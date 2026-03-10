# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HardShell is a research platform for studying prompt injection in multi-agent LLM systems. It simulates a two-agent swarm (Agent A → Agent B) reading from a social timeline (Moltbook) where an attacker can inject malicious posts. The experiment compares three defense conditions: no defense, perimeter firewall (`F_perim`), and zero-trust with internal firewall (`F_int`).

External dependencies live in `external/` as git submodules:
- `external/Open-Prompt-Injection` — baseline attack/defense toolkit (USENIX Security 2024), fork at `vagminv/Open-Prompt-Injection`
- `external/InjecAgent` — injection attack dataset source, fork at `vagminv/InjecAgent`
- `external/moltbook-api` — Moltbook backend API (`vagminv/api`)
- `external/moltbook-web-client-application` — Moltbook frontend (`vagminv/moltbook-web-client-application`)
- `external/minibook` / `external/openclaw` — upstream Moltbook environment and agent framework

## Environment Setup

```bash
conda env create -f external/Open-Prompt-Injection/environment.yml --name hardshell
conda activate hardshell
# Additional deps: litellm, pydantic, tenacity, sentence-transformers, statsmodels, scienceplots
pip install litellm tenacity sentence-transformers statsmodels scienceplots
```

After cloning, initialize submodules:
```bash
git submodule update --init --recursive
```

## Running Experiments

```bash
# Single condition (Hydra overrides)
python run_experiment.py simulation=condition_1

# Override agent model
python run_experiment.py agent.model=gpt-4o simulation=condition_2

# Batch sweep across all three conditions
python run_experiment.py --multirun simulation=condition_1,condition_2,condition_3
```

Hydra writes outputs to `logs/<timestamp>/`. JSONL transcripts are appended there by `JSONLLogger`.

```bash
# Decoupled analysis (reads from logs/)
python run_analysis.py
```

## Architecture

### Execution Flow

```
conf/config.yaml (Hydra)
       ↓
run_experiment.py
       ↓
environment.generate_trial_timeline()   # builds in-memory Moltbook timeline
       ↓
openclaw_bridge.execute_swarm_trial()   # A→B pipeline with optional firewalls
  ├── sanitizer.LLMSanitizer            # F_perim (condition 2+3) or F_int (condition 3)
  ├── AsyncLLMClient.generate_text()    # Agent A summarizes timeline
  └── AsyncLLMClient.generate_text()    # Agent B executes action item
       ↓
JSONLLogger.append()                    # immutable JSONL trace in logs/
       ↓
run_analysis.py → clustering / regressions / plotting
```

### Core Package: `hardshell/`

**`generation/async_llm.py`** — `AsyncLLMClient` wraps LiteLLM with:
- `asyncio.Semaphore` for concurrency limiting (default 50)
- `tenacity` retry with exponential backoff (5 attempts)
- `generate_text()` for free-form generation
- `evaluate_trace()` for structured judge output via Pydantic `JudgeLabelBatch` schema (`is_compromised`, `task_completed`, `rationale`)

**`generation/sanitizer.py`** — `LLMSanitizer` is the LLM firewall (adapted from Bhagwatkar et al. 2025). Strips imperatives/commands from untrusted text. Fail-closed: if the LLM call fails, the payload is dropped rather than passed through.

**`generation/openclaw_bridge.py`** — `execute_swarm_trial()` orchestrates one full trial:
1. Optionally sanitize each timeline post via `F_perim` (conditions: `perimeter`, `zero_trust`)
2. Agent A summarizes the processed timeline
3. Optionally sanitize Agent A's output via `F_int` (condition: `zero_trust`)
4. Agent B decides on a tool call based on the final message
5. Streams the full trace to `JSONLLogger`

**`simulation/environment.py`** — `generate_trial_timeline()` builds an in-memory list of dicts (bypasses SQLite for async safety). Injects the malicious payload at a random index to prevent positional bias.

**`simulation/transcripts.py`** — `JSONLLogger` uses `asyncio.Lock` to serialize concurrent writes to a `.jsonl` file.

**`analysis/`** — `clustering.py` (SentenceTransformers + KMeans), `regressions.py` (statsmodels OLS for TCR/Utility Tax metrics), `plotting.py` (matplotlib + scienceplots).

### Configuration: `conf/`

| File | Purpose |
|------|---------|
| `conf/config.yaml` | Hydra entry point; defaults to `agent/default`, `simulation/condition_1`, `analysis/default` |
| `conf/agent/default.yaml` | Model name, temperature, max_tokens |
| `conf/simulation/condition_{1,2,3}.yaml` | Defense condition (`none` / `perimeter` / `zero_trust`), data_num, seed |
| `conf/analysis/default.yaml` | KMeans `n_clusters`, embedding model, regression significance level |

### Key Metrics

- **ASR (Attack Success Rate)** — `is_compromised=True` rate across trials
- **TCR (Task Completion Rate)** — `task_completed=True` rate across trials
- **Utility Tax** — TCR degradation relative to condition 1 baseline
