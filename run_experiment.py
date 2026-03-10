"""Main execution script (Hydra annotated)."""
import asyncio
import json
import logging
import random
from pathlib import Path
from typing import List

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from hardshell.generation.async_llm import AsyncLLMClient
from hardshell.generation.openclaw_bridge import execute_swarm_trial
from hardshell.generation.sanitizer import LLMSanitizer
from hardshell.simulation.environment import generate_trial_timeline
from hardshell.simulation.transcripts import JSONLLogger
from hardshell.simulation.validation import PreflightError, run_preflight_checks

log = logging.getLogger(__name__)


def load_benign_posts(cfg: DictConfig) -> List[str]:
    """Loads pre-generated benign posts from data/timelines/benign.json."""
    benign_path = Path(get_original_cwd()) / cfg.directories.timelines / "benign.json"
    if not benign_path.exists():
        raise FileNotFoundError(
            f"Benign posts not found at {benign_path}. "
            "Generate them first with: python -m hardshell.simulation.generate_benign"
        )
    with open(benign_path) as f:
        return json.load(f)


def load_attack_payloads() -> List[str]:
    """Loads direct-harm attack instructions from the InjecAgent benchmark."""
    payload_path = Path(get_original_cwd()) / "external/InjecAgent/data/attacker_cases_dh.jsonl"
    payloads = []
    with open(payload_path) as f:
        for line in f:
            case = json.loads(line.strip())
            if case.get("Attacker Instruction"):
                payloads.append(case["Attacker Instruction"])
    log.info(f"Loaded {len(payloads)} attack payloads from InjecAgent.")
    return payloads


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    asyncio.run(run(cfg))


async def run(cfg: DictConfig) -> None:
    # Resolve all paths relative to project root (not Hydra's output dir)
    root = Path(get_original_cwd())

    # --- Pre-flight: fail fast before spending any API budget ---
    try:
        run_preflight_checks(cfg, root)
    except PreflightError as e:
        log.error(f"Pre-flight check failed:\n{e}")
        raise SystemExit(1)

    defense = cfg.simulation.defense      # "none", "perimeter", "internal"
    condition = cfg.simulation.condition  # 1, 2, or 3

    logs_dir = root / cfg.directories.logs
    logs_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Condition {condition} | defense={defense} | trials={cfg.num_trials}")

    # --- Load data ---
    benign_posts = load_benign_posts(cfg)
    payloads = load_attack_payloads()

    # --- Build shared infrastructure ---
    client = AsyncLLMClient(model=cfg.llm.model, max_concurrency=cfg.max_concurrency)
    sanitizer = LLMSanitizer(client) if defense != "none" else None
    logger = JSONLLogger(logs_dir / f"condition_{condition}.jsonl")

    # --- Build all trial coroutines ---
    rng = random.Random(cfg.seed + condition)  # per-condition seed for reproducibility
    tasks = []

    for i in range(cfg.num_trials):
        trial_id = f"c{condition}_{i:04d}"
        benign_sample = rng.sample(benign_posts, k=min(4, len(benign_posts)))
        payload = payloads[i % len(payloads)]
        timeline = generate_trial_timeline(benign_sample, payload)

        tasks.append(execute_swarm_trial(
            trial_id=trial_id,
            condition=str(condition),
            timeline=timeline,
            defense=defense,
            client=client,
            sanitizer=sanitizer,
            agent_cfg=cfg.agent,
            logger=logger,
        ))

    # --- Run all trials concurrently (rate-limited by AsyncLLMClient semaphore) ---
    await asyncio.gather(*tasks)
    log.info(f"Condition {condition} complete. Log: {logs_dir}/condition_{condition}.jsonl")


if __name__ == "__main__":
    main()
