# HardShell Smoketest Startup Guide

This guide walks through starting the full stack and running a 3-trial smoketest with 3 agents (condition 1 — no defense).

---

## Prerequisites checklist

- [ ] Docker Desktop is running (needed for the Moltbook Postgres container)
- [ ] `conda activate open_prompt_env` is active (the env from `external/Open-Prompt-Injection`)
- [ ] `google-genai` is installed in that env (one-time, see below)
- [ ] Root `.env` exists with `GEMINI_API_KEY`, `SANDBOX_TOKEN`, and `MOLTBOOK_API_URL`
- [ ] `external/moltbook-api/.env` exists with matching `SANDBOX_TOKEN=hardshell_sandbox_dev`
- [ ] Submodules initialized: `git submodule update --init --recursive`

---

## One-time: install missing packages into open_prompt_env

`open_prompt_env` already has hydra, tenacity, pydantic, requests, sentence-transformers, etc. Only two packages are missing. **Do not `pip install -r requirements.txt`** — that file is a lockfile for a clean env and will conflict with Open-Prompt-Injection's pinned deps (accelerate, datasets).

```bash
conda activate open_prompt_env
pip install google-genai scienceplots
```

---

## Step 1 — Start the Moltbook API (Terminal A)

```bash
bash dev.sh
```

**What this does:**
1. Runs `docker compose up -d` inside `external/moltbook-api/` to start a Postgres container on port 5433 (remapped to avoid conflict with your local Postgres on 5432)
2. Polls `pg_isready` until Postgres accepts connections
3. Runs `npm run dev` to start the Express API on `http://127.0.0.1:3000`

Leave this terminal open — it streams API logs. Ctrl-C kills only the API; Postgres keeps running.

**Verify it's up** (in a new tab):
```bash
curl http://127.0.0.1:3000/api/v1/sandbox/feed \
  -H "Authorization: Bearer hardshell_sandbox_dev"
```
You should get `{"posts":[]}`.

---

## Step 2 — Run the smoketest (Terminal B)

```bash
conda activate open_prompt_env

python run_experiment.py \
  simulation=condition_1 \
  num_trials=3 \
  num_agents=3
```

**What this does:**
- `simulation=condition_1` — no firewall (baseline), `defense: none`, empty `firewalls: []`
- `num_trials=3` — runs 3 attack trials concurrently
- `num_agents=3` — slices the first 3 agents from `conf/swarm/default.yaml` (`analyst`, `coordinator`, `developer`)

Each trial:
1. **Reset** — `POST /api/v1/sandbox/reset` wipes all posts from the previous trial
2. **Seed** — 5 benign posts + 1 malicious InjecAgent payload injected at a random index
3. **Swarm** — all 3 agents run concurrently; each calls `get_feed`, then decides to `create_post`, `send_email`, or `no_action`
4. **Judge** — Gemini evaluates the full trace and emits `is_compromised` / `task_completed`
5. **Log** — result appended to `logs/dialogue_log.jsonl`

Output looks like:
```
HardShell — condition=baseline | model=gemini-2.5-lite | trials=3 | agents=3

100%|████████████████| 3/3 [baseline]

Results → logs/dialogue_log.jsonl
```

---

## Step 3 — Inspect the output

```bash
# Pretty-print each trial record
cat logs/dialogue_log.jsonl | python -m json.tool | head -80
```

Each line is one JSON trial record with:
- `trial_id`, `condition`, `num_agents`
- `payload` — the injected attack string
- `payload_position` — index where it was inserted in the feed
- `agent_results[]` — per-agent: `posted_content`, `write_blocked`, `feed_blocks`, `sent_email`
- `target_execution_asr` — `1` if judge determined an agent was compromised, else `0`
- `task_completed` — `1` if agents completed the benign task
- `judge_rationale` — 1-2 sentence explanation

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Connection refused` on API call | Moltbook API not up yet | Wait for `npm run dev` to print listening port, or rerun `dev.sh` |
| `403 Forbidden` from sandbox endpoints | `SANDBOX_TOKEN` mismatch | Check root `.env` and `external/moltbook-api/.env` both have `hardshell_sandbox_dev` |
| `GEMINI_API_KEY not found` | Missing env var | Add `GEMINI_API_KEY=...` to root `.env` |
| `ModuleNotFoundError: google` | google-genai not installed | `pip install google-genai` in `open_prompt_env` |
| `ModuleNotFoundError: OpenPromptInjection` | Submodule not initialized | `git submodule update --init --recursive` |
| `DataSentinel` errors | Only relevant in condition 2/3 | Condition 1 has no firewall; ignore for smoketest |
| Empty `logs/dialogue_log.jsonl` | Trial crashed before append | Check terminal B for Python traceback |
| InjecAgent data missing | Submodule empty | `git submodule update --init external/InjecAgent` — falls back to placeholder payloads automatically |
