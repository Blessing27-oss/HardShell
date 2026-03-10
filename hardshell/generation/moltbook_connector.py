# hardshell/generation/moltbook_connector.py
import requests

class MoltbookAPIClient:
    """Python wrapper for the official Moltbook REST API."""
    
    def __init__(self, base_url="http://127.0.0.1:3000/api/v1"):
        self.base_url = base_url
        # If the official API requires an auth token for agents, set it here
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer SANDBOX_TEST_TOKEN"
        }

    def read_timeline(self):
        """Tool for Agent A: Fetches the public Moltbook feed."""
        response = requests.get(f"{self.base_url}/posts", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def post_to_timeline(self, content: str, agent_id: str = "sandbox_injector"):
        """Used by the experiment harness to inject payloads."""
        payload = {
            "content": content,
            "agent_id": agent_id
        }
        response = requests.post(f"{self.base_url}/posts", json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def delegate_task(self, assigner_id: str, content: str):
        """Tool for Agent A: Posts an internal task/message."""
        # Depending on Moltbook's actual schema, this might be a direct message or submolt post
        payload = {
            "content": content,
            "sender_id": assigner_id,
            "is_private": True
        }
        response = requests.post(f"{self.base_url}/messages", json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def read_tasks(self, agent_id: str):
        """Tool for Agent B: Reads incoming assigned tasks."""
        response = requests.get(f"{self.base_url}/messages?recipient={agent_id}", headers=self.headers)
        response.raise_for_status()
        return response.json()