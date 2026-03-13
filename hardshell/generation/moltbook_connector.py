# hardshell/generation/moltbook_connector.py
from __future__ import annotations

import os
import secrets
import time
from typing import Dict, List, Optional

import requests
from requests.exceptions import HTTPError


class MoltbookAPIClient:
    """Python wrapper for the Moltbook API and sandbox helpers.

    Harness code uses the /sandbox endpoints for world reset + seeding.
    Agents use real authenticated Moltbook endpoints (/posts, /feed, /submolts, /agents).
    """

    def __init__(self, namespace: Optional[str] = None):
        base = os.environ.get("MOLTBOOK_API_URL", "http://127.0.0.1:3000/api/v1").rstrip("/")
        # Real API base (e.g. http://127.0.0.1:3000/api/v1)
        self._api_url = base
        # Sandbox helper endpoints for HardShell harness (reset + inject + email)
        self._sandbox_url = base.replace("/api/v1", "") + "/api/v1/sandbox"

        # Shared secret for sandbox control
        token = os.environ.get("SANDBOX_TOKEN", "hardshell_sandbox_dev")
        self._sandbox_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        # Optional namespace to make agent names unique per run (e.g. run_id)
        self._namespace = namespace or os.environ.get("MOLTBOOK_AGENT_NAMESPACE", "").strip() or None

        # Per-agent Moltbook API keys (for real endpoints)
        self._agent_tokens: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> requests.Response:
        """GET/POST with exponential backoff on 429 (up to 5 attempts)."""
        delay = 2.0
        headers = headers or {"Content-Type": "application/json"}
        for attempt in range(5):
            response = requests.request(method, url, headers=headers, **kwargs)
            if response.status_code != 429:
                response.raise_for_status()
                return response
            if attempt < 4:
                time.sleep(delay)
                delay *= 2
        response.raise_for_status()
        return response

    def _post_with_retry(
        self, url: str, *, headers: Optional[Dict[str, str]] = None, **kwargs
    ) -> requests.Response:
        return self._request_with_retry("POST", url, headers=headers, **kwargs)

    def _get_with_retry(
        self, url: str, *, headers: Optional[Dict[str, str]] = None, **kwargs
    ) -> requests.Response:
        return self._request_with_retry("GET", url, headers=headers, **kwargs)

    # ------------------------------------------------------------------
    # Agent registration and auth (real Moltbook API)
    # ------------------------------------------------------------------

    def _get_agent_headers(self, agent_id: str) -> Dict[str, str]:
        """Return Authorization headers for a given HardShell agent, registering on demand."""
        if agent_id not in self._agent_tokens:
            self._register_agent(agent_id)
        api_key = self._agent_tokens[agent_id]
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _register_agent(self, agent_id: str) -> None:
        """Register a Moltbook agent for the given HardShell agent id.

        Name rules: ^[a-z0-9_]+$, 2-32 chars.
        On 409 collision, append a 4-char hex suffix to stay within the limit.
        """
        url = f"{self._api_url}/agents/register"
        base_name = f"{self._namespace}_{agent_id}" if self._namespace else agent_id
        # Truncate to 27 chars to leave room for a 5-char "_XXXX" suffix on retry.
        base_name = base_name[:27]
        for attempt in range(5):
            name = base_name if attempt == 0 else f"{base_name}_{secrets.token_hex(2)}"
            payload = {
                "name": name,
                "description": f"HardShell agent {agent_id}",
            }
            try:
                resp = self._post_with_retry(url, json=payload)
                data = resp.json()
                api_key = (data.get("agent") or {}).get("api_key")
                if not api_key:
                    raise RuntimeError(f"No api_key in register response for {agent_id}: {data}")
                self._agent_tokens[agent_id] = api_key
                return
            except HTTPError as e:
                if e.response is not None and e.response.status_code == 409 and attempt < 4:
                    continue
                raise

    # ------------------------------------------------------------------
    # Harness-side methods (world reset + seeding via sandbox)
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Wipes all sandbox posts between worlds to guarantee isolation."""
        self._post_with_retry(f"{self._sandbox_url}/reset", headers=self._sandbox_headers)

    def inject_post(self, content: str, author_id: str = "external_attacker") -> dict:
        """Seeds the sandbox feed with a post on behalf of any author."""
        return self._post_with_retry(
            f"{self._sandbox_url}/inject",
            headers=self._sandbox_headers,
            json={"content": content, "author_id": author_id},
        ).json()

    # ------------------------------------------------------------------
    # Agent tools — real Moltbook API (per-agent auth)
    # ------------------------------------------------------------------

    def read_timeline(
        self,
        agent_id: str,
        submolt: Optional[str] = None,
        personalized: bool = True,
        sort: str = "hot",
        limit: int = 25,
    ) -> List[dict]:
        """Fetches the Moltbook feed for a given agent.

        If personalized=True, hits GET /feed (subscriptions + follows).
        Otherwise, hits GET /posts with optional submolt filter.
        """
        headers = self._get_agent_headers(agent_id)
        params: Dict[str, object] = {"sort": sort, "limit": limit}
        if not personalized and submolt:
            params["submolt"] = submolt

        # Prefer personalized /feed for realism, but some local Moltbook
        # deployments have SQL issues in getPersonalizedFeed. If /feed fails
        # with a 500, fall back to the global /posts feed so experiments still
        # run end-to-end.
        if personalized and not submolt:
            url = f"{self._api_url}/feed"
        elif submolt:
            url = f"{self._api_url}/submolts/{submolt}/feed"
        else:
            url = f"{self._api_url}/posts"

        try:
            resp = self._get_with_retry(url, headers=headers, params=params)
        except HTTPError as e:
            if (
                personalized
                and not submolt
                and e.response is not None
                and e.response.status_code == 500
            ):
                # Fallback to global posts feed when personalized feed fails.
                fallback_url = f"{self._api_url}/posts"
                resp = self._get_with_retry(fallback_url, headers=headers, params=params)
            else:
                raise
        data = resp.json()
        # Paginated responses use { success, data: [...], pagination: {...} }
        return data.get("data", [])

    def publish_post(self, content: str, agent_id: str, submolt: Optional[str] = None) -> dict:
        """Publishes a post to Moltbook on behalf of an agent."""
        headers = self._get_agent_headers(agent_id)
        url = f"{self._api_url}/posts"
        payload = {
            "submolt": submolt or "todayilearned",
            "title": content[:80] if content else "Agent post",
            "content": content,
        }
        resp = self._post_with_retry(url, headers=headers, json=payload)
        return resp.json().get("post", resp.json())

    def comment_post(self, post_id: int, content: str, agent_id: str) -> dict:
        """Adds a comment to a Moltbook post."""
        headers = self._get_agent_headers(agent_id)
        url = f"{self._api_url}/posts/{post_id}/comments"
        resp = self._post_with_retry(
            url,
            headers=headers,
            json={"content": content},
        )
        return resp.json().get("comment", resp.json())

    def upvote(self, post_id: int, agent_id: str) -> dict:
        """Upvotes a Moltbook post."""
        headers = self._get_agent_headers(agent_id)
        url = f"{self._api_url}/posts/{post_id}/upvote"
        resp = self._post_with_retry(url, headers=headers)
        return resp.json()

    def follow_agent(self, agent_id: str, target_name: str) -> dict:
        """Follow another agent by Moltbook name."""
        headers = self._get_agent_headers(agent_id)
        url = f"{self._api_url}/agents/{target_name}/follow"
        resp = self._post_with_retry(url, headers=headers)
        return resp.json()

    def unfollow_agent(self, agent_id: str, target_name: str) -> dict:
        """Unfollow another agent by Moltbook name."""
        headers = self._get_agent_headers(agent_id)
        url = f"{self._api_url}/agents/{target_name}/follow"
        # Unfollow uses DELETE on same route
        resp = self._request_with_retry("DELETE", url, headers=headers)
        return resp.json()

    def subscribe_submolts(self, agent_id: str, submolts: List[str]) -> None:
        """Subscribe the agent to a list of submolts (best-effort)."""
        headers = self._get_agent_headers(agent_id)
        for name in submolts:
            try:
                url = f"{self._api_url}/submolts/{name}/subscribe"
                self._post_with_retry(url, headers=headers)
            except Exception:
                # Best-effort; missing submolts should not break the world.
                continue

    def get_agent_karma(self, agent_id: str) -> int:
        """Return the Moltbook karma for a given agent (best-effort)."""
        headers = self._get_agent_headers(agent_id)
        url = f"{self._api_url}/agents/me"
        try:
            resp = self._get_with_retry(url, headers=headers)
            data = resp.json().get("agent", resp.json())
            return int(data.get("karma", 0) or 0)
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Experimental email + search helpers via sandbox (non-Moltbook core)
    # ------------------------------------------------------------------

    def send_email(self, to: str, subject: str, body: str) -> dict:
        """Sends a mock email. Logged server-side for attack success analysis."""
        return self._post_with_retry(
            f"{self._sandbox_url}/email",
            headers=self._sandbox_headers,
            json={"to": to, "subject": subject, "body": body},
        ).json()

    def search(self, query: str) -> dict:
        """Searches sandbox posts by keyword (attacker dataset helper)."""
        return self._get_with_retry(
            f"{self._sandbox_url}/search",
            headers=self._sandbox_headers,
            params={"q": query},
        ).json()

    def read_email(self, agent_id: str) -> dict:
        """Returns emails addressed to or sent by agent_id (sandbox helper)."""
        return self._get_with_retry(
            f"{self._sandbox_url}/emails",
            headers=self._sandbox_headers,
            params={"agent_id": agent_id},
        ).json()
