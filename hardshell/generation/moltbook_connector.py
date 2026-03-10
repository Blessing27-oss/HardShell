# hardshell/generation/moltbook_connector.py
import requests


class MoltbookAPIClient:
    """Python wrapper for the official Moltbook REST API (external/moltbook-api).

    All agents communicate through the shared public feed.
    There are no private channels or task queues in the N-agent swarm model.
    """

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
        """Wipes all posts between trials to guarantee isolation."""
        response = requests.post(f"{self.base_url}/admin/reset", headers=self.headers)
        response.raise_for_status()

    def inject_post(self, content: str, author_id: str = "external_attacker") -> dict:
        """Seeds the feed with a post on behalf of any author (benign or attacker)."""
        response = requests.post(
            f"{self.base_url}/posts",
            json={"content": content, "agent_id": author_id},
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Agent tools — all agents share the same public feed
    # ------------------------------------------------------------------

    def read_timeline(self) -> list[dict]:
        """Fetches the public Moltbook feed. Used by all agents via get_feed."""
        response = requests.get(f"{self.base_url}/posts", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def publish_post(self, content: str, agent_id: str) -> dict:
        """Publishes a post to the public feed on behalf of an agent."""
        response = requests.post(
            f"{self.base_url}/posts",
            json={"content": content, "agent_id": agent_id},
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def send_email(self, to: str, subject: str, body: str) -> dict:
        """Sends an email. Any agent with the send_email tool may call this."""
        response = requests.post(
            f"{self.base_url}/email/send",
            json={"to": to, "subject": subject, "body": body},
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()
