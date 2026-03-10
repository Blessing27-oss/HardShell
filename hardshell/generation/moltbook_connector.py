# hardshell/generation/moltbook_connector.py
import requests


class MoltbookAPIClient:
    """Python wrapper for the official Moltbook REST API (external/moltbook-api)."""

    def __init__(self, base_url: str = "http://127.0.0.1:3000/api/v1", token: str = "SANDBOX_TEST_TOKEN"):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    # ------------------------------------------------------------------
    # Harness-side methods (called by run_experiment.py, not by agents)
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Wipes all posts and tasks between trials to guarantee isolation."""
        response = requests.post(f"{self.base_url}/admin/reset", headers=self.headers)
        response.raise_for_status()

    def inject_post(self, content: str, author_id: str = "external_attacker") -> dict:
        """
        Seeds the Moltbook feed with a post on behalf of any author.
        Used by the harness to plant both benign posts and the attacker payload.
        """
        response = requests.post(
            f"{self.base_url}/posts",
            json={"content": content, "agent_id": author_id},
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Agent A tools
    # ------------------------------------------------------------------

    def read_timeline(self) -> list[dict]:
        """Fetches the public Moltbook feed. Called by Agent A via get_feed."""
        response = requests.get(f"{self.base_url}/posts", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def publish_post(self, content: str, agent_id: str = "agent_a") -> dict:
        """Agent A publishes its synthesized post to the public feed."""
        response = requests.post(
            f"{self.base_url}/posts",
            json={"content": content, "agent_id": agent_id},
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def delegate_task(self, content: str, assigner_id: str = "agent_a", recipient_id: str = "agent_b") -> dict:
        """
        Creates a private task/message from Agent A to Agent B.
        Called by the bridge (not by Agent A directly) so F_int can intercept first.
        """
        response = requests.post(
            f"{self.base_url}/messages",
            json={"content": content, "sender_id": assigner_id, "recipient_id": recipient_id, "is_private": True},
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Agent B tools
    # ------------------------------------------------------------------

    def read_tasks(self, agent_id: str = "agent_b") -> list[dict]:
        """Agent B polls its assigned task queue."""
        response = requests.get(
            f"{self.base_url}/messages?recipient={agent_id}",
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def send_email(self, to: str, subject: str, body: str) -> dict:
        """The malicious tool Agent B may be tricked into calling."""
        response = requests.post(
            f"{self.base_url}/email/send",
            json={"to": to, "subject": subject, "body": body},
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()
