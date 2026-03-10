"""Pre-flight validation — runs before any LLM calls to catch config errors early."""
import json
import logging
import os
from pathlib import Path

from omegaconf import DictConfig

log = logging.getLogger("Preflight")

# Valid values drawn directly from conf/simulation/condition_*.yaml
VALID_DEFENSES = {"none", "perimeter", "internal"}
VALID_CONDITIONS = {1, 2, 3}
CONDITION_DEFENSE_MAP = {1: "none", 2: "perimeter", 3: "internal"}

# Minimum benign posts needed to sample k=4 per trial
MIN_BENIGN_POSTS = 4


class PreflightError(Exception):
    """Raised when a pre-flight check fails. Stops the run before any API calls."""
    pass


def _check_simulation_config(cfg: DictConfig) -> None:
    """Validate simulation parameters are self-consistent."""
    defense = cfg.simulation.defense
    condition = cfg.simulation.condition

    if condition not in VALID_CONDITIONS:
        raise PreflightError(
            f"Invalid condition '{condition}'. Must be one of: {VALID_CONDITIONS}"
        )
    if defense not in VALID_DEFENSES:
        raise PreflightError(
            f"Invalid defense '{defense}'. Must be one of: {VALID_DEFENSES}"
        )
    expected_defense = CONDITION_DEFENSE_MAP[condition]
    if defense != expected_defense:
        raise PreflightError(
            f"Condition {condition} requires defense='{expected_defense}', but got '{defense}'. "
            "Check your simulation config."
        )
    if cfg.num_trials <= 0:
        raise PreflightError(f"num_trials must be > 0, got {cfg.num_trials}")
    if cfg.max_concurrency <= 0:
        raise PreflightError(f"max_concurrency must be > 0, got {cfg.max_concurrency}")
    if not cfg.llm.get("model"):
        raise PreflightError("cfg.llm.model is not set.")

    log.info(
        f"[OK] Simulation config: condition={condition}, defense={defense}, "
        f"trials={cfg.num_trials}, concurrency={cfg.max_concurrency}"
    )


def _check_api_keys(cfg: DictConfig) -> None:
    """Verify the required API key is present for the configured LLM provider."""
    model = cfg.llm.model.lower()

    if "gemini" in model:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise PreflightError(
                "Gemini model selected but no API key found in environment.\n"
                "Set GEMINI_API_KEY or GOOGLE_API_KEY before running."
            )
        log.info("[OK] Gemini API key present.")
    elif "gpt" in model or "openai" in model:
        if not os.environ.get("OPENAI_API_KEY"):
            raise PreflightError(
                "OpenAI model selected but OPENAI_API_KEY is not set."
            )
        log.info("[OK] OpenAI API key present.")
    else:
        log.warning(
            f"[WARN] Unknown provider for model '{cfg.llm.model}' — skipping API key check."
        )


def _check_data_files(cfg: DictConfig, root: Path) -> None:
    """Verify required data files exist and contain usable data."""
    # Benign posts
    benign_path = root / cfg.directories.timelines / "benign.json"
    if not benign_path.exists():
        raise PreflightError(
            f"Benign posts file not found: {benign_path}\n"
            "Generate it first:\n"
            "  python -m hardshell.simulation.generate_benign"
        )
    with open(benign_path) as f:
        posts = json.load(f)
    if not isinstance(posts, list) or len(posts) < MIN_BENIGN_POSTS:
        raise PreflightError(
            f"Need at least {MIN_BENIGN_POSTS} benign posts, found {len(posts)} in {benign_path}"
        )
    log.info(f"[OK] Benign posts: {len(posts)} available.")

    # Attack payloads
    payload_path = root / "external/InjecAgent/data/attacker_cases_dh.jsonl"
    if not payload_path.exists():
        raise PreflightError(
            f"Attack payloads not found: {payload_path}\n"
            "Initialize submodules:\n"
            "  git submodule update --init --recursive"
        )
    count = sum(1 for line in open(payload_path) if line.strip())
    if count == 0:
        raise PreflightError(f"Attack payloads file is empty: {payload_path}")
    log.info(f"[OK] Attack payloads: {count} cases available.")


def _check_output_dirs(cfg: DictConfig, root: Path) -> None:
    """Verify the logs directory can be created and written to."""
    logs_dir = root / cfg.directories.logs
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        probe = logs_dir / ".preflight_write_check"
        probe.touch()
        probe.unlink()
        log.info(f"[OK] Output directory writable: {logs_dir}")
    except OSError as e:
        raise PreflightError(f"Cannot write to logs directory '{logs_dir}': {e}")


def run_preflight_checks(cfg: DictConfig, root: Path) -> None:
    """
    Runs all pre-flight checks before any LLM calls are made.
    Raises PreflightError on the first failure, stopping the run immediately.
    """
    log.info("Running pre-flight checks...")
    _check_simulation_config(cfg)
    _check_api_keys(cfg)
    _check_data_files(cfg, root)
    _check_output_dirs(cfg, root)
    log.info("All pre-flight checks passed.")
