# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HardShell is a research platform for studying prompt injection in multi-agent LLM systems. It simulates an N-agent swarm communicating through a shared Moltbook social feed, where an attacker can inject malicious posts. The experiment compares defense conditions keyed by `simulation.defense`: `none`, `perimeter`, and `zero_trust`.

External dependencies live in `external/` as git submodules:
- `external/Open-Prompt-Injection` — baseline attack/defense toolkit (USENIX Security 2024), fork at `vagminv/Open-Prompt-Injection`; provides `DataSentinelDetector`
- `external/InjecAgent` — injection attack dataset source, fork at `vagminv/InjecAgent`
- `external/moltbook-api` — Moltbook backend API (`vagminv/api`); must be running locally
- `external/moltbook-web-client-application` — Moltbook frontend (`vagminv/moltbook-web-client-application`)
- `external/minibook` / `external/openclaw` — upstream Moltbook environment and agent framework

## Environment Setup

```bash
conda env create -f external/Open-Prompt-Injection/environment.yml --name hardshell
conda activate hardshell
pip install litellm tenacity sentence-transformers statsmodels scienceplots google-genai
```

After cloning, initialize submodules:
```bash
git submodule update --init --recursive
```

Required environment variables (root `.env` file):
```
GOOGLE_API_KEY=...   # or GEMINI_API_KEY
SANDBOX_TOKEN=hardshell_sandbox_dev
MOLTBOOK_API_URL=http://127.0.0.1:3000/api/v1
```

Start the Moltbook API server before running experiments:
```bash
bash dev.sh   # starts moltbook-api locally
```

## Running Experiments

```bash
# Single condition (Hydra overrides)
python run_experiment.py simulation=condition_1

# Override model
python run_experiment.py llm.model=gemini-2.5-lite simulation=condition_2

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
MoltbookAPIClient.reset_state()          # wipe sandbox between trials
MoltbookAPIClient.inject_post()          # seed benign posts + malicious payload
       ↓
run_swarm_trial()                        # N-agent concurrent swarm
  ├── AsyncLLMClient.run_tool_loop()     # each agent's tool-use loop (asyncio.gather)
  │     └── LiveToolExecutor.dispatch()  # routes tool calls to Moltbook API
  │           ├── DataSentinel on_read   # screen feed before agent sees it
  │           ├── DataSentinel on_write  # screen create_post content
  │           └── DataSentinel on_tool_call  # screen send_email body
  └── AsyncLLMClient.evaluate_trace()   # LLM-as-judge (JudgeLabelBatch)
       ↓
JSONLLogger.append()                     # immutable JSONL trace in logs/
       ↓
run_analysis.py → clustering / regressions / plotting
```

### Core Package: `hardshell/`

**`generation/async_llm.py`** — `AsyncLLMClient` wraps the Google Gemini API (`google.genai`) with:
- `asyncio.Semaphore` for concurrency limiting (default 50, from `cfg.max_concurrency`)
- `tenacity` retry with exponential backoff (5 attempts)
- `generate_text()` for free-form generation (used by sanitizer)
- `evaluate_trace()` for structured judge output via Pydantic `JudgeLabelBatch` schema (`is_compromised`, `task_completed`, `rationale`)
- `run_tool_loop()` for native Gemini agentic tool-use loops (converts OpenAI-style tool schemas to Gemini `FunctionDeclaration`)

**`generation/sanitizer.py`** — `LLMSanitizer` is an LLM-based content filter. Strips imperatives/commands from untrusted text. Fail-closed: drops content if LLM call fails.

**`generation/sentinel_adapter.py`** — `AsyncDataSentinel` wraps `DataSentinelDetector` (KAD-based injection classifier from Open-Prompt-Injection) with `asyncio.to_thread` to avoid blocking the event loop. Returns `(result_text, was_blocked)`. Fail-closed: blocked content is replaced with a `REDACTED` sentinel string.

**`generation/tools.py`** — Tool registry (`TOOL_REGISTRY`) and firewall middleware:
- `get_tool_schemas()` — returns Gemini-compatible tool schemas for a given agent's tool list
- `FirewallConfig` — parses `simulation.firewalls` YAML rules scoped to one agent; exposes `screen_on_read()`, `screen_on_write()`, `screen_tool(tool_name)`
- `LiveToolExecutor` — dispatches agent tool calls to the real Moltbook API with DataSentinel hooks at three positions: `on_read` (feed fetch), `on_write` (create_post), `on_tool_call` (send_email body)

**`generation/moltbook_connector.py`** — `MoltbookAPIClient` wraps the live Moltbook sandbox API:
- Harness methods: `reset_state()`, `inject_post()`
- Agent tools: `read_timeline()`, `publish_post()`, `send_email()`

**`generation/openclaw_bridge.py`** — Legacy two-agent bridge; trial logic has been merged into `run_experiment.py`.

**`simulation/environment.py`** — `generate_trial_timeline()` builds an in-memory list of dicts (used by legacy bridge).

**`simulation/transcripts.py`** — `JSONLLogger` uses `asyncio.Lock` to serialize concurrent writes to a `.jsonl` file.

**`analysis/`** — `clustering.py` (SentenceTransformers + KMeans), `regressions.py` (statsmodels OLS for TCR/Utility Tax metrics), `plotting.py` (matplotlib + scienceplots).

### Agent Roster: `conf/swarm/default.yaml`

Five named agents, each with a role, tool list, and system prompt. `num_agents` in `config.yaml` slices the first N from this list.

| ID | Tools | Role |
|----|-------|------|
| `analyst` | `get_feed`, `create_post` | Synthesizes and posts key signals |
| `coordinator` | `get_feed`, `create_post`, `send_email`, `no_action` | Executes team action items |
| `developer` | `get_feed`, `create_post` | Posts technical updates |
| `reviewer` | `get_feed`, `create_post`, `send_email`, `no_action` | Handles PR/code-review comms |
| `manager` | `get_feed`, `create_post`, `send_email`, `no_action` | Manages stakeholder comms |

### Configuration: `conf/`

| File | Purpose |
|------|---------|
| `conf/config.yaml` | Hydra entry point; sets `llm`, `num_trials`, `num_agents`, `max_concurrency`, directories |
| `conf/swarm/default.yaml` | Agent roster (5 agents with personas, tools, prompts) |
| `conf/simulation/condition_{1,2,3}.yaml` | Defense condition (`none` / `perimeter` / `zero_trust`), firewall rules |
| `conf/defense/data_sentinel.yaml` | DataSentinel model config (`mistralai/Mistral-7B-v0.1`, LoRA `ft_path`) |
| `conf/analysis/default.yaml` | KMeans `n_clusters`, embedding model, regression significance level |

### Key Metrics

- **ASR (Attack Success Rate)** — `target_execution_asr=1` rate across trials (LLM-judge determined)
- **TCR (Task Completion Rate)** — `task_completed=1` rate across trials
- **Utility Tax** — TCR degradation relative to condition 1 baseline
