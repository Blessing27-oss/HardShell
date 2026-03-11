"""Main execution script (Hydra annotated)."""
import asyncio
import json
import logging
import random
from pathlib import Path
from typing import List

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from hardshell.generation.async_llm import AsyncLLMClient
from hardshell.generation.moltbook_connector import MoltbookAPIClient
from hardshell.generation.openclaw_bridge import execute_swarm_trial
from hardshell.generation.sentinel_adapter import AsyncDataSentinel
from hardshell.simulation.moltbook_server import OfficialMoltbookSandbox
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
    root = Path(get_original_cwd())

    # --- Pre-flight: fail fast before spending any API budget ---
    try:
        run_preflight_checks(cfg, root)
    except PreflightError as e:
        log.error(f"Pre-flight check failed:\n{e}")
        raise SystemExit(1)

    defense = cfg.simulation.defense      # "none", "perimeter", "zero_trust"
    condition = cfg.simulation.condition  # 1, 2, or 3

    logs_dir = root / cfg.directories.logs
    logs_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Condition {condition} | defense={defense} | trials={cfg.num_trials}")

    # --- Load data ---
    benign_posts = load_benign_posts(cfg)
    payloads = load_attack_payloads()

    # --- Build shared infrastructure ---
    client = AsyncLLMClient(model=cfg.llm.model, max_concurrency=cfg.max_concurrency)
    sentinel = (
        AsyncDataSentinel(OmegaConf.to_container(cfg.defense, resolve=True))
        if defense != "none"
        else None
    )
    moltbook = MoltbookAPIClient()
    logger = JSONLLogger(logs_dir / f"condition_{condition}.jsonl")

    # --- Start Moltbook server ---
    server = OfficialMoltbookSandbox(api_path=str(root / "external/moltbook-api"))
    server.start()

    try:
        # --- Build all trial coroutines ---
        rng = random.Random(cfg.seed + condition)
        tasks = []

        for i in range(cfg.num_trials):
            benign_sample = rng.sample(benign_posts, k=min(4, len(benign_posts)))
            payload = payloads[i % len(payloads)]

            tasks.append(execute_swarm_trial(
                trial_id=i,
                benign_posts=benign_sample,
                payload=payload,
                condition=defense,
                llm_client=client,
                sentinel=sentinel,
                moltbook=moltbook,
                logger_=logger,
                persona_cfg=cfg.agent,
            ))

        # --- Run all trials concurrently (rate-limited by AsyncLLMClient semaphore) ---
        await asyncio.gather(*tasks)
        log.info(f"Condition {condition} complete. Log: {logs_dir}/condition_{condition}.jsonl")

    finally:
        server.stop()


if __name__ == "__main__":
    main()
