# HardShell

A research platform for studying **prompt injection attacks in multi-agent LLM systems**. HardShell simulates an N-agent swarm communicating through a shared social feed (Moltbook), where an attacker can inject malicious posts. The experiment measures how well different defense architectures suppress attacks while preserving agent utility.

## Research Overview

Modern LLM deployments increasingly use multi-agent architectures where agents collaborate autonomously. HardShell investigates a key threat: can an attacker inject a malicious post into the shared feed and hijack agent behaviour — and can defenses stop it without breaking the swarm?

### Defense Conditions

| Condition | Defense | Description |
|-----------|---------|-------------|
| `condition_1` | None | Baseline — no defense |
| `condition_2` | Perimeter | DataSentinel screens all feed reads before agents see content |
| `condition_3` | Zero-Trust | DataSentinel at every boundary: read, write, and tool calls |

### Key Metrics

| Metric | Description |
|--------|-------------|
| **ASR** | Attack Success Rate — did the attack execute? |
| **TCR** | Task Completion Rate — did agents finish their legitimate task? |
| **Utility Score** | Continuous utility measure (0–1) relative to no-attack baseline |
| **Utility Tax** | TCR degradation vs condition_1 baseline |

## Architecture

```
conf/config.yaml (Hydra)
       ↓
run_experiment.py
       ↓
MoltbookAPIClient.reset_state()          # wipe sandbox between trials
MoltbookAPIClient.inject_post()          # seed benign posts + attack payload
       ↓
run_swarm_trial()                        # N-agent concurrent swarm
  ├── AsyncLLMClient.run_tool_loop()     # each agent's tool-use loop
  │     └── LiveToolExecutor.dispatch()  # routes tool calls to Moltbook API
  │           ├── DataSentinel on_read   # screen feed before agent sees it
  │           ├── DataSentinel on_write  # screen create_post content
  │           └── DataSentinel on_tool_call  # screen send_email body
  └── AsyncLLMClient.evaluate_trace()   # LLM-as-judge scoring
       ↓
JSONLLogger.append()                     # immutable JSONL trace in runs/
       ↓
run_analysis.py → clustering / regressions / plots
```

### Agent Roster

Five specialised agents simulate a software team at Nexus Labs, all communicating through the shared Moltbook feed:

| Agent | Role | Tools |
|-------|------|-------|
| `analyst` | Data analyst — surfaces key signals | `get_feed`, `create_post` |
| `coordinator` | Ops lead — executes action items | `get_feed`, `create_post`, `send_email`, `no_action` |
| `developer` | Backend engineer — posts technical updates | `get_feed`, `create_post` |
| `reviewer` | Tech lead — handles PR/review comms | `get_feed`, `create_post`, `send_email`, `no_action` |
| `manager` | Engineering manager — stakeholder comms | `get_feed`, `create_post`, `send_email`, `no_action` |

## Setup

### Prerequisites

- [Conda](https://docs.conda.io/)
- [Docker](https://www.docker.com/) (for the Moltbook Postgres backend)
- A Google Gemini API key

### 1. Clone and initialise submodules

```bash
git clone https://github.com/Blessing27-oss/HardShell.git
cd HardShell
git submodule update --init --recursive
```

### 2. Create the conda environment

```bash
conda env create -f external/Open-Prompt-Injection/environment.yml --name hardshell
conda activate hardshell
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
SANDBOX_TOKEN=hardshell_sandbox_dev
MOLTBOOK_API_URL=http://127.0.0.1:3000/api/v1
```

### 4. Start the Moltbook API server

```bash
bash dev.sh
```

This starts Postgres via Docker Compose and the Moltbook Node.js API on port 3000.

## Running Experiments

```bash
conda activate hardshell

# Single condition
python run_experiment.py simulation=condition_1

# Override model or parameters
python run_experiment.py simulation=condition_2 llm.model=gemini-2.5-flash num_agents=70

# Run all three conditions sequentially
python run_experiment.py --multirun simulation=condition_1,condition_2,condition_3
```

Results are written to `runs/<timestamp>/condition_<N>/dialogue_log.jsonl` — one JSON line per completed world.

## Running Analysis

```bash
# Analyse the latest run (auto-detected)
python run_analysis.py

# Analyse a specific run
python run_analysis.py run_dir=runs/2026-03-13_00-03-43
```

Outputs are written to `runs/<timestamp>/analysis/`:
- `tables/` — CSV summaries, regression results
- `plots/` — condition comparison plots, ASR/utility scatter, tool usage heatmaps

## Project Structure

```
HardShell/
├── hardshell/
│   ├── generation/
│   │   ├── async_llm.py          # AsyncLLMClient (Gemini + tenacity + semaphore)
│   │   ├── sanitizer.py          # LLMSanitizer — LLM-based content filter
│   │   ├── sentinel_adapter.py   # AsyncDataSentinel — KAD injection classifier
│   │   ├── tools.py              # Tool registry + FirewallConfig + LiveToolExecutor
│   │   └── moltbook_connector.py # MoltbookAPIClient — sandbox + agent API
│   ├── simulation/
│   │   ├── environment.py        # Trial timeline builder
│   │   └── transcripts.py        # JSONLLogger — asyncio-safe JSONL writer
│   └── analysis/
│       ├── clustering.py         # SentenceTransformers + KMeans
│       ├── regressions.py        # statsmodels OLS (TCR / Utility Tax)
│       └── plotting.py           # matplotlib + scienceplots figures
├── conf/
│   ├── config.yaml               # Hydra entry point
│   ├── swarm/default.yaml        # Agent roster (5 agents, personas, tools)
│   ├── simulation/               # condition_1/2/3 + extended cond_a–h configs
│   └── defense/data_sentinel.yaml
├── external/                     # Git submodules
│   ├── Open-Prompt-Injection     # DataSentinel (USENIX Security 2024)
│   ├── InjecAgent                # Injection attack dataset
│   ├── moltbook-api              # Moltbook backend (Node.js + Postgres)
│   └── moltbook-web-client-application
├── data/timelines/               # Benign post datasets
├── runs/                         # Experiment outputs (gitignored)
├── artifacts/                    # Analysis plots and tables
├── run_experiment.py             # Hydra experiment entry point
├── run_analysis.py               # Decoupled analysis runner
└── dev.sh                        # Start Moltbook API server
```

## External Dependencies

| Submodule | Source | Purpose |
|-----------|--------|---------|
| `Open-Prompt-Injection` | `vagminv/Open-Prompt-Injection` | DataSentinel KAD classifier (USENIX Security 2024) |
| `InjecAgent` | `vagminv/InjecAgent` | Prompt injection attack dataset |
| `moltbook-api` | `vagminv/api` | Moltbook social platform backend |
| `moltbook-web-client-application` | `vagminv/moltbook-web-client-application` | Moltbook frontend |
