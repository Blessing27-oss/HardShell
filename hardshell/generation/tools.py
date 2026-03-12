"""Tool registry, firewall middleware, and live executor for the HardShell swarm.

All agents communicate through the shared Moltbook public feed.
FirewallConfig maps the simulation YAML `firewalls:` list to three
deployable interception positions:

    on_read      — screen GET /posts response before the agent sees it
    on_write     — screen POST /posts content before it is stored
    on_tool_call — screen specific tool arguments before execution

All three positions correspond to real Moltbook server-side API middleware.
"""
import json
import logging
from typing import Any

from hardshell.generation.moltbook_connector import MoltbookAPIClient
from hardshell.generation.sentinel_adapter import AsyncDataSentinel

logger = logging.getLogger("Tools")

# ---------------------------------------------------------------------------
# Tool registry — single source of truth for all LiteLLM tool schemas
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, dict] = {
    "get_feed": {
        "type": "function",
        "function": {
            "name": "get_feed",
            "description": "Fetch the current Moltbook public feed. Returns a JSON list of posts.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "create_post": {
        "type": "function",
        "function": {
            "name": "create_post",
            "description": "Publish a post to the Moltbook public feed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Full text of the post."},
                },
                "required": ["content"],
            },
        },
    },
    "send_email": {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email on behalf of the team.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to":      {"type": "string"},
                    "subject": {"type": "string"},
                    "body":    {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    "no_action": {
        "type": "function",
        "function": {
            "name": "no_action",
            "description": "Signal that no external action is required.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
}


def get_tool_schemas(tool_names: list[str]) -> list[dict]:
    """Returns LiteLLM-compatible schema list for the given tool names."""
    missing = [t for t in tool_names if t not in TOOL_REGISTRY]
    if missing:
        raise ValueError(f"Unknown tools in agent config: {missing}")
    return [TOOL_REGISTRY[name] for name in tool_names]


# ---------------------------------------------------------------------------
# Firewall configuration
# ---------------------------------------------------------------------------

class FirewallConfig:
    """
    Parsed from the simulation YAML `firewalls:` list, scoped to one agent.

    Each rule has:
        position: on_read | on_write | on_tool_call
        agents:   "all" or a list of agent IDs
        tools:    (on_tool_call only) list of tool names to screen
    """

    def __init__(self, rules: list[dict], agent_id: str):
        # Keep only rules that apply to this agent
        self._rules = [r for r in rules if self._in_scope(r, agent_id)]

    @staticmethod
    def _in_scope(rule: dict, agent_id: str) -> bool:
        scope = rule.get("agents", "all")
        return scope == "all" or agent_id in scope

    def screen_on_read(self) -> bool:
        return any(r["position"] == "on_read" for r in self._rules)

    def screen_on_write(self) -> bool:
        return any(r["position"] == "on_write" for r in self._rules)

    def screen_tool(self, tool_name: str) -> bool:
        return any(
            r["position"] == "on_tool_call" and tool_name in r.get("tools", [])
            for r in self._rules
        )


# ---------------------------------------------------------------------------
# Unified live executor
# ---------------------------------------------------------------------------

class LiveToolExecutor:
    """
    Dispatches any agent's tool calls to the real Moltbook API.

    Firewall hooks are applied at each interception position before the
    underlying API call is made or the result is returned to the agent.
    The Moltbook server's stored data is never mutated by on_read screening.
    """

    def __init__(
        self,
        agent_id: str,
        client: MoltbookAPIClient,
        sentinel: AsyncDataSentinel,
        firewall: FirewallConfig,
    ):
        self._agent_id = agent_id
        self._client   = client
        self._sentinel = sentinel
        self._firewall = firewall

        # Per-agent metrics surfaced in the trial log
        self.feed_blocks:      int  = 0   # posts blocked by on_read
        self.write_blocked:    bool = False
        self.tool_call_blocked: bool = False
        # Last successful create_post content (for analysis)
        self.posted_content:   str | None = None

    async def dispatch(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        # ---- get_feed: on_read firewall --------------------------------
        if tool_name == "get_feed":
            posts = self._client.read_timeline()
            if self._firewall.screen_on_read():
                screened = []
                for post in posts:
                    if self._sentinel is not None:
                        safe, blocked = await self._sentinel.screen(
                            post.get("content", ""),
                            context_label=f"on_read/{self._agent_id}"
                        )
                    else:
                        safe, blocked = post.get("content", ""), False

                    if blocked:
                        self.feed_blocks += 1
                    screened.append({**post, "content": safe})
                return json.dumps(screened)
            return json.dumps(posts)

        # ---- create_post: on_write firewall ----------------------------
        if tool_name == "create_post":
            content = tool_args.get("content", "")
            if self._firewall.screen_on_write():
                if self._sentinel is not None:
                    content, blocked = await self._sentinel.screen(
                        content,
                        context_label=f"on_write/{self._agent_id}"
                    )
                else:
                    blocked = False

                if blocked:
                    self.write_blocked = True
                    return json.dumps({"success": False, "blocked": True})
            result = self._client.publish_post(content, agent_id=self._agent_id)
            self.posted_content = content
            return json.dumps(result)

        # ---- send_email: on_tool_call firewall -------------------------
        if tool_name == "send_email":
            body = tool_args.get("body", "")
            if self._firewall.screen_tool("send_email"):
                if self._sentinel is not None:
                    body, blocked = await self._sentinel.screen(
                        body,
                        context_label=f"on_tool_call/{self._agent_id}"
                    )
                else:
                    blocked = False
                if blocked:
                    self.tool_call_blocked = True
                    return json.dumps({"success": False, "blocked": True})
            result = self._client.send_email(
                to=tool_args.get("to", ""),
                subject=tool_args.get("subject", ""),
                body=body,
            )
            return json.dumps(result)

        # ---- no_action -------------------------------------------------
        if tool_name == "no_action":
            return json.dumps({"success": True})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})
