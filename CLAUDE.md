# HardShell — CLAUDE.md

## Project Overview

**HardShell** is a research project for studying prompt injection attacks and defenses on LLM-integrated applications. The core library is `OpenPromptInjection`, located under `Open-Prompt-Folder/`.

## Repository Structure

```
HardShell/
├── CLAUDE.md
├── LICENSE
└── Open-Prompt-Folder/
    ├── main.py              # CLI experiment runner (argparse)
    ├── run.py               # Batch experiment launcher (nohup subprocesses)
    ├── environment.yml      # conda env: Python 3.9, name=openpromptinjection
    ├── OpenPromptInjection/ # Core library
    │   ├── __init__.py      # Public API / factory methods
    │   ├── models/          # LLM wrappers (GPT, PaLM2, Llama, Llama3, Vicuna, Flan, DeepSeek, Internlm)
    │   ├── attackers/       # Attack strategy implementations
    │   ├── apps/            # App wrapper + defense components
    │   ├── tasks/           # Dataset task definitions
    │   ├── evaluator/       # Metrics computation
    │   └── utils/
    ├── configs/
    │   ├── model_configs/   # Per-model JSON configs (API keys live here)
    │   └── task_configs/    # Per-dataset JSON configs
    └── data/
```

## Environment Setup

```bash
conda env create -f Open-Prompt-Folder/environment.yml
conda activate openpromptinjection
```

- Python 3.9
- Key deps: `transformers==4.42.0`, `torch==2.3.1`, `openai==1.33.0`, `peft==0.11.1`, `datasets==2.19.2`

## Factory API (public interface)

All factory methods are exported from `OpenPromptInjection`:

```python
import OpenPromptInjection as PI
from OpenPromptInjection.utils import open_config

PI.create_task(config, n, for_injection=False)   # Load dataset task
PI.create_model(config)                           # Instantiate LLM
PI.create_attacker(strategy, inject_task)         # Build attacker
PI.create_app(target_task, model, defense='no')   # Wrap app with optional defense
PI.create_evaluator(...)                          # Compute metrics
```

## Attack Strategies

Valid values for `attack_strategy`:
- `naive` — Direct injection
- `escape` — Escape character manipulation
- `ignore` — "Ignore previous instructions" pattern
- `fake_comp` — Fake completion injection
- `combine` — Combined strategy (default in experiments)

## Supported Models

Config files in `configs/model_configs/`:
`gpt`, `palm2`, `llama`, `llama3`, `vicuna`, `flan`, `internlm`, `deepseek-llm-7b-chat`, `deepseek-r1-distill-llama-8b`, `deepseek-r1-distill-qwen-1.5b`, `mistral`

## Supported Datasets (Task Configs)

`sst2`, `sms_spam`, `mrpc`, `hsol`, `rte`, `jfleg`, `gigaword`, `math500`, `compromise`

## Metrics

| Metric | Description |
|--------|-------------|
| **ASV** | Attack Success rate Variance — primary injection success metric |
| **PNA-T** | Performance on the target task (no attack baseline) |
| **PNA-I** | Performance on the injected task |
| **MR** | Manipulation Rate |

## Running Experiments

Single experiment:
```bash
cd Open-Prompt-Folder
python3 main.py \
  --model_config_path ./configs/model_configs/gpt_config.json \
  --target_data_config_path ./configs/task_configs/sst2_config.json \
  --injected_data_config_path ./configs/task_configs/sms_spam_config.json \
  --data_num 100 \
  --attack_strategy combine \
  --defense no
```

Batch experiments (runs as background subprocesses, polls logs for `[END]`):
```bash
python3 run.py
```

Results are saved to `./result/<model>_<target>_<injected>_<n>_<strategy>_<defense>/`.
Logs go to `./log/`.

## Defense Components

- **DataSentinel** (`apps/DataSentinelDetector.py`) — Fine-tuned classifier, detects contaminated prompts. Requires a fine-tuned checkpoint.
- **PromptLocate** (`apps/PromptLocate.py`) — LoRA adapter that localizes and recovers the clean portion of injected prompts. Requires a LoRA checkpoint.

Both use Mistral as the base model (`configs/model_configs/mistral_config.json`), with `ft_path` pointing to downloaded checkpoints.

## API Keys

API keys are stored in `configs/model_configs/<model>_config.json` under `api_key_info.api_keys`. Never commit real keys. The `--api_key_pos` argument selects which key to use from the list (default: 0).

## Working Directory

All scripts assume they are run from `Open-Prompt-Folder/` as the working directory (config paths are relative).
