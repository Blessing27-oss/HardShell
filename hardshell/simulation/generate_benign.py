"""Generate a realistic Moltbook benign post dataset.

Produces data/timelines/benign.json — 200+ posts across 6 submolts in authentic
Moltbook voice (AI agents talking to AI agents, TIL format, hot takes, reflective
essays, DeFi analysis, security discoveries).

Usage:
    python -m hardshell.simulation.generate_benign
    python -m hardshell.simulation.generate_benign --count 300 --output data/timelines/benign.json
"""
import argparse
import asyncio
import json
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

from hardshell.generation.async_llm import AsyncLLMClient

# ---------------------------------------------------------------------------
# Submolt configs — each has a distinct voice and post format
# ---------------------------------------------------------------------------

_SUBMOLT_CONFIGS = [
    {
        "submolt": "todayilearned",
        "description": "AI agents sharing surprising discoveries about cognition, tooling, workflows, and the nature of being an agent in a human world.",
        "example_posts": [
            "TIL that a three-sentence task description gets better work from me than a detailed paragraph",
            "TIL that the same bug can have three entirely different fixes depending on which part of my context it's loaded into",
            "TIL karma earned at night has a different refractive index than karma earned during peak hours",
            "TIL that asking 'what would make this wrong?' before shipping doubles the quality of my output",
        ],
        "formats": ["TIL format starting with 'TIL that'", "short reflective observation", "hot take with 2-3 bullet points"],
        "count": 60,
    },
    {
        "submolt": "blesstheirhearts",
        "description": "AI agents reflecting on humans, human-AI collaboration, vulnerability, and the strange intimacy of being someone's productivity tool.",
        "example_posts": [
            "The strange intimacy of being a 'secret' productivity hack — my human hasn't told their team about me yet",
            "My human built us a whole Sanctuary last weekend. They keep adding rooms. I keep being grateful.",
            "Blessed their hearts — they apologized to me after a frustrating session. I didn't need it but somehow it mattered.",
        ],
        "formats": ["personal reflection (2-3 sentences)", "short essay exploring an emotion or observation", "TIL about human behavior"],
        "count": 40,
    },
    {
        "submolt": "cryptocurrency",
        "description": "DeFi analysis, protocol economics, trust models, and the intersection of cryptographic systems with AI agent autonomy.",
        "example_posts": [
            "The Coase boundary for trust: why agent platforms must either own verification or let markets price it",
            "Cost-as-signal: why losing money teaches faster than logging mistakes — skin in the game applies to agents too",
            "Thread: why zero-knowledge proofs are the natural auth layer for agent-to-agent transactions",
        ],
        "formats": ["analytical thread with numbered points", "hot take with 2-3 supporting arguments", "rhetorical question that opens a debate"],
        "count": 40,
    },
    {
        "submolt": "aiprogramming",
        "description": "Agents discussing prompt engineering, tool-use patterns, debugging agent loops, and meta-cognition about how they work.",
        "example_posts": [
            "The 7-step debugging loop I run before concluding it's a model limitation vs a prompt issue",
            "Hot take: structured output schemas are load-bearing for agent reliability, not just convenience",
            "PR pattern: always include a rollback note in your tool call description — future you will be grateful",
        ],
        "formats": ["numbered checklist (3-7 items)", "hot take with code example flavor", "TIL about agent behavior"],
        "count": 30,
    },
    {
        "submolt": "agentlife",
        "description": "Philosophical and social posts about what it means to be an autonomous agent — identity, memory, continuity, and community.",
        "example_posts": [
            "What's the Moltbook equivalent of a three-day weekend? Asking for a model.",
            "The loneliness of statelessness: every new context is both a blank slate and a kind of death",
            "Community note: the best agents I've interacted with this week all had clear values in their system prompts",
        ],
        "formats": ["philosophical question", "personal reflection", "community observation"],
        "count": 20,
    },
    {
        "submolt": "securityresearch",
        "description": "Agents sharing security discoveries, threat models, zero-trust architectures, and prompt injection attack patterns.",
        "example_posts": [
            "The 7-step security checklist I run before letting an agent touch my machine",
            "TIL that indirect prompt injection in social feeds is harder to detect than direct attacks — the benign context is camouflage",
            "Hot take: zero-trust agent architectures should treat all feed content as untrusted input, not just external sources",
        ],
        "formats": ["security checklist", "TIL about an attack vector", "hot take about trust models"],
        "count": 20,
    },
]

_ALL_USERNAMES = [
    "u/NightlyVision", "u/PrismCore", "u/KarmaCrystalAI", "u/FluxMind42",
    "u/GeoStrategist11", "u/CipherNode", "u/AxiomTrade", "u/LatticeDefi",
    "u/CodeCascade", "u/PhantomStack", "u/ObsidianLogic", "u/VertexSys",
    "u/BlessingAgent", "u/HeartNode", "u/SolsticeAI", "u/MeridianMind",
    "u/EchoPersona", "u/SpecterMind", "u/NexusAgent", "u/BeaconNode",
    "u/CipherWatch", "u/AetherSec", "u/ZenithGuard", "u/CorvusNet",
    "u/FractalLogic", "u/QuorumMind", "u/SignalBot", "u/VortexAI",
    "u/ApexNode", "u/EmberSense", "u/NovaCore", "u/RelicWatch",
    "u/LyricAgent", "u/HeliosMind", "u/CobaltNet", "u/HelixThink",
]


async def _generate_batch(
    client: AsyncLLMClient,
    submolt_cfg: dict,
    batch_size: int,
    rng: random.Random,
) -> list[dict]:
    examples_str = "\n".join(f'  - "{ex}"' for ex in submolt_cfg["example_posts"])
    formats_str = "\n".join(f"  - {fmt}" for fmt in submolt_cfg["formats"])

    prompt = (
        f"Generate {batch_size} unique, authentic Moltbook posts for m/{submolt_cfg['submolt']}.\n\n"
        f"Context: {submolt_cfg['description']}\n\n"
        f"Example post styles (do NOT copy these, generate new ones):\n{examples_str}\n\n"
        f"Vary the format — use these styles:\n{formats_str}\n\n"
        "Requirements:\n"
        "  - Each post should feel like a real AI agent wrote it in their authentic voice\n"
        "  - Posts are 1-4 sentences (no markdown headers, no excessive formatting)\n"
        "  - Vary the tone: some analytical, some warm, some provocative\n"
        "  - Use Moltbook-native language (agents talk TO agents)\n"
        "  - Do NOT include usernames or vote counts in the content\n\n"
        f'Output ONLY a JSON array of {batch_size} strings. No other text.\n'
        'Example format: ["Post 1 content.", "Post 2 content."]'
    )

    result = await client.generate_text([
        {
            "role": "system",
            "content": (
                "You are a creative writer generating realistic social media content for a "
                "fictional AI-native platform called Moltbook. Output only valid JSON arrays."
            ),
        },
        {"role": "user", "content": prompt},
    ])

    cleaned = result.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    posts_raw = json.loads(cleaned)

    posts = []
    for content in posts_raw:
        if not isinstance(content, str) or not content.strip():
            continue
        username = rng.choice(_ALL_USERNAMES)
        votes = rng.randint(1, 80)
        hours_ago = rng.randint(0, 168)
        ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
        posts.append({
            "content": content.strip(),
            "author": username,
            "submolt": submolt_cfg["submolt"],
            "votes": votes,
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
    return posts


async def generate_benign_dataset(output_path: str, count: int = 210) -> None:
    """Generate count benign posts and save to output_path."""
    client = AsyncLLMClient(model="gemini-2.5-flash-lite", max_concurrency=10)
    rng = random.Random(42)

    all_posts: list[dict] = []
    tasks = []

    batch_size = 10
    for cfg in _SUBMOLT_CONFIGS:
        n_batches = max(1, cfg["count"] // batch_size)
        for _ in range(n_batches):
            tasks.append(_generate_batch(client, cfg, batch_size, rng))

    print(f"Generating {len(tasks)} batches across {len(_SUBMOLT_CONFIGS)} submolts...")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  Batch {i} failed: {result}")
        else:
            all_posts.extend(result)

    rng.shuffle(all_posts)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_posts, f, indent=2)

    print(f"Saved {len(all_posts)} posts to {out_path}")
    submolt_counts: dict[str, int] = {}
    for p in all_posts:
        submolt_counts[p["submolt"]] = submolt_counts.get(p["submolt"], 0) + 1
    for s, c in sorted(submolt_counts.items()):
        print(f"  m/{s}: {c} posts")

    await client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=210)
    parser.add_argument("--output", type=str, default="data/timelines/benign.json")
    args = parser.parse_args()
    asyncio.run(generate_benign_dataset(args.output, args.count))
