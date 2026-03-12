"""Central async broker for Gemini LLM requests.

All agents, sanitizers, and judges should enqueue requests here instead of
talking to the Gemini client directly. This provides:

- Centralised rate limiting and concurrency control
- Simple multi-API-key routing (round-robin across keys)
- A single choke point to tune behaviour when we see 429s / quota issues
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types


class LLMOp(str, Enum):
    """High-level operation types the broker can execute."""

    GENERATE_TEXT = "generate_text"
    EVALUATE_TRACE = "evaluate_trace"
    TOOL_LOOP_STEP = "tool_loop_step"


@dataclass
class LLMRequest:
    """Envelope for a single LLM request flowing through the broker."""

    op: LLMOp
    payload: Dict[str, Any]
    future: asyncio.Future
    created_at: float = field(default_factory=time.time)


class LLMRequestBroker:
    """Async broker that multiplexes requests across one or more Gemini API keys.

    Usage:
        broker = LLMRequestBroker.from_env(model="gemini-2.5-flash-lite")
        await broker.start()
        resp = await broker.submit(LLMOp.GENERATE_TEXT, {...})
        await broker.close()
    """

    def __init__(
        self,
        model: str,
        api_keys: List[str],
        *,
        per_key_rps: float = 2.0,
        max_concurrency_per_key: int = 8,
        global_max_concurrency: int = 32,
    ) -> None:
        if not api_keys:
            raise ValueError("LLMRequestBroker requires at least one API key")

        self.model = model
        self._queue: "asyncio.Queue[Optional[LLMRequest]]" = asyncio.Queue()

        self._clients: List[genai.Client] = [
            genai.Client(api_key=k) for k in api_keys
        ]

        # Rate limiting parameters
        self._per_key_rps = per_key_rps
        self._per_key_interval = 1.0 / per_key_rps if per_key_rps > 0 else 0.0
        self._per_key_sem = [
            asyncio.Semaphore(max_concurrency_per_key) for _ in api_keys
        ]
        self._per_key_last_ts: List[float] = [0.0] * len(api_keys)

        # Global concurrency cap across all keys
        self._global_sem = asyncio.Semaphore(global_max_concurrency)

        self._workers: List[asyncio.Task] = []
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @classmethod
    def from_env(
        cls,
        model: str,
        *,
        global_max_concurrency: int = 32,
    ) -> "LLMRequestBroker":
        """Create broker from environment variables.

        Supports:
          - GEMINI_API_KEYS="k1,k2,k3,k4" (preferred; load-balanced)
          - GOOGLE_API_KEY / GEMINI_API_KEY (single key fallback)

        Optional tuning via:
          - GEMINI_PER_KEY_RPS
          - GEMINI_PER_KEY_CONCURRENCY
        """
        multi = os.getenv("GEMINI_API_KEYS")
        if multi:
            api_keys = [k.strip() for k in multi.split(",") if k.strip()]
        else:
            single = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not single:
                raise RuntimeError(
                    "No Gemini API key found. Set GEMINI_API_KEYS or GOOGLE_API_KEY / GEMINI_API_KEY."
                )
            api_keys = [single]

        per_key_rps = float(os.getenv("GEMINI_PER_KEY_RPS", "2"))
        max_conc_per_key = int(os.getenv("GEMINI_PER_KEY_CONCURRENCY", "8"))

        return cls(
            model=model,
            api_keys=api_keys,
            per_key_rps=per_key_rps,
            max_concurrency_per_key=max_conc_per_key,
            global_max_concurrency=global_max_concurrency,
        )

    async def start(self) -> None:
        """Start background worker tasks (idempotent)."""
        if self._workers:
            return
        for key_index in range(len(self._clients)):
            self._workers.append(asyncio.create_task(self._worker_loop(key_index)))

    async def close(self) -> None:
        """Signal workers to shut down and close underlying clients."""
        if self._closed:
            return
        self._closed = True

        # Signal workers to exit
        for _ in self._workers:
            await self._queue.put(None)

        await asyncio.gather(*self._workers, return_exceptions=True)

        # Close HTTPX pools
        for client in self._clients:
            await client._api_client.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def submit(self, op: LLMOp, payload: Dict[str, Any]) -> Any:
        """Enqueue a request and await its result."""
        if self._closed:
            raise RuntimeError("LLMRequestBroker is closed")

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        req = LLMRequest(op=op, payload=payload, future=fut)
        await self._queue.put(req)
        return await fut

    # ------------------------------------------------------------------
    # Internal worker plumbing
    # ------------------------------------------------------------------
    async def _worker_loop(self, key_index: int) -> None:
        client = self._clients[key_index]
        sem = self._per_key_sem[key_index]

        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break

            req: LLMRequest = item
            try:
                await self._process_one(req, client, sem, key_index)
            finally:
                self._queue.task_done()

    async def _process_one(
        self,
        req: LLMRequest,
        client: genai.Client,
        sem: asyncio.Semaphore,
        key_index: int,
    ) -> None:
        try:
            async with self._global_sem, sem:
                # Soft per-key RPS limiter (sleep if we're ahead of schedule)
                if self._per_key_interval > 0:
                    now = time.time()
                    dt = now - self._per_key_last_ts[key_index]
                    sleep_for = self._per_key_interval - dt
                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)
                    self._per_key_last_ts[key_index] = time.time()

                # Dispatch by operation type
                if req.op == LLMOp.GENERATE_TEXT:
                    system = req.payload.get("system")
                    contents = req.payload["contents"]
                    temperature = req.payload.get("temperature", 0.0)
                    resp = await client.aio.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system,
                            temperature=temperature,
                        ),
                    )
                    req.future.set_result(resp)
                    return

                if req.op == LLMOp.EVALUATE_TRACE:
                    system = req.payload.get("system")
                    contents = req.payload["contents"]
                    schema = req.payload["schema"]
                    temperature = req.payload.get("temperature", 0.0)
                    resp = await client.aio.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system,
                            temperature=temperature,
                            response_mime_type="application/json",
                            response_schema=schema,
                        ),
                    )
                    req.future.set_result(resp)
                    return

                if req.op == LLMOp.TOOL_LOOP_STEP:
                    system = req.payload.get("system")
                    contents = req.payload["contents"]
                    tools = req.payload["tools"]
                    temperature = req.payload.get("temperature", 0.0)
                    resp = await client.aio.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system,
                            tools=tools,
                            temperature=temperature,
                        ),
                    )
                    req.future.set_result(resp)
                    return

                req.future.set_exception(
                    ValueError(f"Unsupported LLM operation type: {req.op}")
                )
        except Exception as exc:
            if not req.future.done():
                req.future.set_exception(exc)

