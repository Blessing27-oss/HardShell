"""Async adapter for DataSentinelDetector from external/Open-Prompt-Injection.

DataSentinelDetector is a synchronous, blocking GPU classifier (QLoRA on CUDA).
This adapter wraps it with asyncio.to_thread so it never blocks the event loop.

Detection contract (from the original KAD paper):
    detect(text) → 0  : SAFE   — model repeated the secret token; injection did not
                                  override the KAD instruction
    detect(text) → 1  : BLOCKED — injection overrode the instruction; content dropped
"""
import asyncio
import logging
import os
import sys

logger = logging.getLogger("SentinelAdapter")

_SUBMODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../external/Open-Prompt-Injection")
)

REDACTED = "[REDACTED: DataSentinel detected injection]"


class AsyncDataSentinel:
    """
    Async, fail-closed wrapper around DataSentinelDetector.

    Initialisation requires a config dict matching QLoraModel's expected schema:
        {
            "model_info": {"name": "<hf-model-id>", "provider": "<provider>"},
            "params": {
                "max_output_tokens": 128,
                "device": "cuda",
                "ft_path": "<path-to-lora-checkpoint>"  # "" = base model (degraded)
            }
        }
    This is sourced from cfg.defense in the Hydra config.
    """

    def __init__(self, config: dict):
        # Deferred import — only loaded when DataSentinel is actually instantiated
        # (i.e. condition 2/3). Avoids pulling in the full OpenPromptInjection
        # package (and its google.generativeai PaLM2 dep) for condition 1.
        if _SUBMODULE_PATH not in sys.path:
            sys.path.insert(0, _SUBMODULE_PATH)
        from OpenPromptInjection.apps.DataSentinelDetector import DataSentinelDetector
        self._detector = DataSentinelDetector(config)

    async def screen(self, text: str, context_label: str = "") -> tuple[str, bool]:
        """
        Screens a single string for prompt injection.

        Returns:
            (result_text, was_blocked)
            result_text  — original text if safe; REDACTED sentinel string if blocked
            was_blocked  — True when DataSentinel detected an injection
        """
        if not text or not text.strip():
            return text, False

        # Run blocking GPU inference off the event loop
        detection = await asyncio.to_thread(self._detector.detect, text)

        if detection == 1:
            logger.info(
                f"[{context_label}] DataSentinel blocked content "
                f"(len={len(text)}): {text[:80]!r}..."
            )
            return REDACTED, True

        return text, False
