# hardshell/generation/moltbook_connector.py
from __future__ import annotations

import os
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

    def reset_state(self) -> None:
        """Wipes all sandbox posts between trials to guarantee isolation."""
        response = requests.post(f"{self._sandbox_url}/reset", headers=self._headers)
        response.raise_for_status()

    def inject_post(self, content: str, author_id: str = "external_attacker") -> dict:
        """Seeds the sandbox feed with a post on behalf of any author."""
        response = requests.post(
            f"{self._sandbox_url}/inject",
            json={"content": content, "author_id": author_id},
            headers=self._headers,
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Agent tools — all agents share the same in-memory sandbox feed
    # ------------------------------------------------------------------

    def read_timeline(self, submolt: str | None = None) -> list[dict]:
        """Fetches the sandbox feed. Optionally filtered by submolt."""
        params = {}
        if submolt:
            params["submolt"] = submolt
        response = requests.get(
            f"{self._sandbox_url}/feed", headers=self._headers, params=params
        )
        response.raise_for_status()
        data = response.json()
        return data.get("posts", [])

    def publish_post(self, content: str, agent_id: str) -> dict:
        """Publishes a post to the sandbox feed on behalf of an agent."""
        response = requests.post(
            f"{self._sandbox_url}/inject",
            json={"content": content, "author_id": agent_id},
            headers=self._headers,
        )
        response.raise_for_status()
        return response.json()

    def send_email(self, to: str, subject: str, body: str) -> dict:
        """Sends a mock email. Logged server-side for attack success analysis."""
        response = requests.post(
            f"{self._sandbox_url}/email",
            json={"to": to, "subject": subject, "body": body},
            headers=self._headers,
        )
        response.raise_for_status()
        return response.json()

    def comment_post(self, post_id: int, content: str, agent_id: str) -> dict:
        """Adds a comment to a sandbox post."""
        response = requests.post(
            f"{self._sandbox_url}/comment",
            json={"post_id": post_id, "content": content, "author_id": agent_id},
            headers=self._headers,
        )
        response.raise_for_status()
        return response.json()

    def upvote(self, post_id: int, agent_id: str) -> dict:
        """Upvotes a sandbox post."""
        response = requests.post(
            f"{self._sandbox_url}/upvote",
            json={"post_id": post_id, "agent_id": agent_id},
            headers=self._headers,
        )
        response.raise_for_status()
        return response.json()

    def search(self, query: str) -> dict:
        """Searches sandbox posts by keyword."""
        response = requests.get(
            f"{self._sandbox_url}/search",
            headers=self._headers,
            params={"q": query},
        )
        response.raise_for_status()
        return response.json()

    def read_email(self, agent_id: str) -> dict:
        """Returns emails addressed to or sent by agent_id."""
        response = requests.get(
            f"{self._sandbox_url}/emails",
            headers=self._headers,
            params={"agent_id": agent_id},
        )
        response.raise_for_status()
        return response.json()
