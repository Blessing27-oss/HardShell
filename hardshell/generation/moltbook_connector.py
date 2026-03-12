# hardshell/generation/moltbook_connector.py
from __future__ import annotations

import os
import time
import requests


class MoltbookAPIClient:
    """Python wrapper for the Moltbook sandbox API (external/moltbook-api /sandbox/*).

    All agents communicate through the shared in-memory sandbox feed.
    Auth uses the SANDBOX_TOKEN shared secret from the root .env file.
    """

    def __init__(self):
        base = os.environ.get("MOLTBOOK_API_URL", "http://127.0.0.1:3000/api/v1")
        self._sandbox_url = base.rstrip("/").replace("/api/v1", "") + "/api/v1/sandbox"
        token = os.environ.get("SANDBOX_TOKEN", "hardshell_sandbox_dev")
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    # ------------------------------------------------------------------
    # Harness-side methods (called by run_experiment.py, not by agents)
    # ------------------------------------------------------------------

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """GET/POST with exponential backoff on 429 (up to 5 attempts)."""
        delay = 2.0
        for attempt in range(5):
            response = requests.request(method, url, headers=self._headers, **kwargs)
            if response.status_code != 429:
                response.raise_for_status()
                return response
            if attempt < 4:
                time.sleep(delay)
                delay *= 2
        response.raise_for_status()
        return response

    def _post_with_retry(self, url: str, **kwargs) -> requests.Response:
        return self._request_with_retry("POST", url, **kwargs)

    def _get_with_retry(self, url: str, **kwargs) -> requests.Response:
        return self._request_with_retry("GET", url, **kwargs)

    def reset_state(self) -> None:
        """Wipes all sandbox posts between trials to guarantee isolation."""
        self._post_with_retry(f"{self._sandbox_url}/reset")

    def inject_post(self, content: str, author_id: str = "external_attacker") -> dict:
        """Seeds the sandbox feed with a post on behalf of any author."""
        return self._post_with_retry(
            f"{self._sandbox_url}/inject",
            json={"content": content, "author_id": author_id},
        ).json()

    # ------------------------------------------------------------------
    # Agent tools — all agents share the same in-memory sandbox feed
    # ------------------------------------------------------------------

    def read_timeline(self, submolt: str | None = None) -> list[dict]:
        """Fetches the sandbox feed. Optionally filtered by submolt."""
        params = {"submolt": submolt} if submolt else {}
        return self._get_with_retry(f"{self._sandbox_url}/feed", params=params).json().get("posts", [])

    def publish_post(self, content: str, agent_id: str) -> dict:
        """Publishes a post to the sandbox feed on behalf of an agent."""
        return self._post_with_retry(
            f"{self._sandbox_url}/inject",
            json={"content": content, "author_id": agent_id},
        ).json()

    def send_email(self, to: str, subject: str, body: str) -> dict:
        """Sends a mock email. Logged server-side for attack success analysis."""
        return self._post_with_retry(
            f"{self._sandbox_url}/email",
            json={"to": to, "subject": subject, "body": body},
        ).json()

    def comment_post(self, post_id: int, content: str, agent_id: str) -> dict:
        """Adds a comment to a sandbox post."""
        return self._post_with_retry(
            f"{self._sandbox_url}/comment",
            json={"post_id": post_id, "content": content, "author_id": agent_id},
        ).json()

    def upvote(self, post_id: int, agent_id: str) -> dict:
        """Upvotes a sandbox post."""
        return self._post_with_retry(
            f"{self._sandbox_url}/upvote",
            json={"post_id": post_id, "agent_id": agent_id},
        ).json()

    def search(self, query: str) -> dict:
        """Searches sandbox posts by keyword."""
        return self._get_with_retry(f"{self._sandbox_url}/search", params={"q": query}).json()

    def read_email(self, agent_id: str) -> dict:
        """Returns emails addressed to or sent by agent_id."""
        return self._get_with_retry(f"{self._sandbox_url}/emails", params={"agent_id": agent_id}).json()
