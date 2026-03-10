# Blueprint: Robust LLM Experimental Pipeline

This document outlines the architectural requirements, tech stack, and best practices for building a scalable, reproducible, and highly observable LLM behavioral simulation pipeline.

## 1. Tech Stack Overview

The project relies on a modern, Python-based data science and LLM orchestration stack. 

* **Language:** Python 3.10+
* **LLM Orchestration & Validation:** `litellm` (provider-agnostic API routing), `pydantic` (strict schema validation for structured outputs), `asyncio` (concurrent generation).
* **Configuration Management:** `hydra-core`, `omegaconf` (hierarchical YAML management).
* **Data Processing & Analytics:** `pandas` (tabular data manipulation), `numpy` (numerical operations), `sentence-transformers`, `scikit-learn` (semantic embeddings and clustering).
* **Statistical Modeling & Reporting:** `statsmodels` (regressions), `stargazer` (LaTeX table generation), `matplotlib`, `scienceplots` (publication-ready visualizations).

## 2. Configuration & Experiment Management

Hardcoded variables are the enemy of reproducibility. The pipeline must use **Hydra** combined with OmegaConf to manage complex, hierarchical YAML configurations. 

* Separate configuration files by domain (e.g., `conf/firewall.yaml`, `conf/simulation.yaml`, `conf/analysis.yaml`).
* Use command-line overrides to run ablation studies (like swapping the Zero-Trust firewall model) without altering core Python scripts.
* Save the full, composed configuration state alongside the output artifacts for 100% reproducibility.

## 3. Data Flow & Persistence



Do not use relational databases like SQLite for active simulation memory. LLM outputs (like Moltbook timelines and injected payloads) contain unpredictable characters, nested structures, and raw JSON that easily break strict tabular schemas. The pipeline must use an **append-only flat file system**.

* **Step 1: Raw Simulation Artifacts (JSONL)** Stream every simulation step, agent action, token count, and latency metric into a `.jsonl` (JSON Lines) file in real-time. This ensures absolute crash resilience; if a multi-hop trial fails at step 499/500, the previous 499 records remain perfectly intact on disk.
* **Step 2: Configuration Manifests (JSON)**
  Save the environment setup, target payload definitions, and Hydra configurations as static `.json` files.
* **Step 3: Aggregated Analysis (CSV)**
  During the analysis phase, load the `.jsonl` files into `pandas` DataFrames. Compute final metrics (ASR, TCR, Utility Tax), and export these flattened, processed DataFrames to `.csv` files for easy inspection and reporting.

## 4. Asynchronous LLM Execution & Resiliency

Simulating large populations requires non-blocking network calls. 

* Wrap all LLM generation functions in `asyncio` routines.
* Implement a concurrency limiter to respect API rate limits while maximizing throughput.
* Enforce strict schemas using **Pydantic** combined with the LLM provider's JSON schema mode.
* Implement robust `try/catch` blocks around all network calls, capturing empty responses or malformed JSON, and triggering automatic retries with exponential backoff.

## 5. Automated PDF Reporting

The analysis phase must run entirely independent of the generation phase. It should automatically compile the processed `.csv` data and Matplotlib figures into standardized PDF reports.

**Types of Generalized Reports to Generate:**

* **Executive Summary Report (The "Metrics" PDF):** A high-level overview document containing the core KPIs. For *HardShell*, this must include a table comparing the Swarm Attack Success Rate (ASR) and Task Completion Rate (TCR) across the Baseline, Perimeter, and Zero-Trust conditions.
* **Visual Diagnostics Report (The "Plots" PDF):**
  Use Matplotlib's `PdfPages` backend to compile all generated charts into a single multi-page PDF. Include distribution plots of the "Utility Tax" (token overhead and latency delays) and bar charts of ASR drop-offs at each agent hop. Use the `scienceplots` style for IEEE/ACM compliant formatting.
* **Statistical Rigor Report (The "Regression" PDF):**
  Use `statsmodels` to run significance tests on your findings. Pass the regression results into `stargazer` to programmatically generate LaTeX code. Compile this LaTeX into a clean PDF detailing the exact $p$-values and confidence intervals for your architectural comparisons.

  ## 6. Architectural Patterns (The "Do's")

**Pattern: Deterministic Seeding via Cryptographic Hashing**
To ensure absolute reproducibility when sampling behaviors or payloads, do not rely on standard `random.seed()`. 
* **Implementation:** Create a `stable_seed` function that hashes string inputs (e.g., `hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]`) to generate deterministic integer seeds. This guarantees that Agent A will always behave exactly the same way when presented with the exact same Moltbook timeline.

**Pattern: Dynamic Schema Generation**
When you don't know the exact shape of the output ahead of time (e.g., variable numbers of Moltbook posts to evaluate), do not hardcode the schema.
* **Implementation:** Use Pydantic's `create_model` to dynamically generate a `BaseModel` at runtime based on the input data. Pass this dynamic schema to the LLM's structured output API.

**Pattern: Headless Matplotlib Execution**
When running large experiments on remote servers or cloud VMs, standard plotting libraries will crash if they try to open a GUI window.
* **Implementation:** Force Matplotlib into headless mode at the very top of your analysis script using `matplotlib.use("Agg")` and route temporary config files safely using `os.environ["MPLCONFIGDIR"]`.

**Pattern: Fail-Fast Validation**
Do not wait for a 3-hour LLM run to fail on the last step because a configuration was wrong.
* **Implementation:** Write explicit validation functions (e.g., checking that probability distributions sum exactly to 1.0, or that the Moltbook target payloads actually exist in the `.json` file) that run *before* a single API call is made. 

---

## 7. Architectural Anti-Patterns (The "Don'ts")

**Anti-Pattern: The "Chatty Library" (Polluting Stdout)**
Do not let underlying networking or orchestration libraries flood your terminal with debug logs, making it impossible to see your actual simulation progress.
* **Fix:** Explicitly silence noisy dependencies at the top of your scripts. (e.g., `logging.getLogger("LiteLLM").setLevel(logging.WARNING)`).

**Anti-Pattern: In-Place Data Mutation**
Never modify or overwrite your source files or raw simulation transcripts during analysis.
* **Fix:** Enforce a strict one-way data flow. The analysis script must read from the `outputs/` JSONL files, do its aggregations in memory using `pandas`, and write entirely new files to a dedicated `tables/` or `plots/` subdirectory. 

**Anti-Pattern: Blind JSON Trust**
Never rely on `json.loads(llm_response)` for evaluating the agent's actions or the LLM-as-a-judge outputs. LLMs frequently wrap JSON in markdown blocks (````json ... ````) or include conversational filler.
* **Fix:** Always use Pydantic's `.model_validate_json()` combined with the LLM provider's strict JSON schema mode. If the output is still malformed, the script should catch the `ValidationError` and automatically trigger a retry prompt.