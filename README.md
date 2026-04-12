---
title: Data Cleaning OpenEnv
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - data-cleaning
---

# 🧹 Data Cleaning — OpenEnv RL Environment

> An OpenEnv-compatible reinforcement learning environment where an LLM agent cleans messy real-world CSV datasets across three difficulty levels.

---

## Why This Environment?

Data cleaning is one of the most time-consuming tasks in data science — analysts spend up to **80% of their time** cleaning and preparing data before any analysis begins. This environment simulates the exact kinds of problems data engineers and analysts face every day:

- Missing values that break downstream models
- Inconsistent formats that cause join failures and parsing errors
- Duplicate records and outliers that corrupt statistics

Training an RL agent on this environment teaches it to **reason about data quality**, apply the right transformation, and receive structured feedback — making it a strong benchmark for real-world agentic capability.

---

## Tasks

Three tasks with clear difficulty progression:

| Task | Problem | Difficulty | Solvable In |
|------|---------|------------|-------------|
| Easy | Fill 3 missing values (age, salary, city) using mean & forward-fill | ⭐ Easy | 1 step |
| Medium | Standardize name casing, phone formats, and date formats | ⭐⭐ Medium | 1 step |
| Hard | Remove duplicate rows + replace numeric outliers with column means | ⭐⭐⭐ Hard | 1 step |

Each task has a **programmatic grader** that scores the agent's output between `0.01` and `0.99` — deterministic, reproducible, and fair.

---

## Action Space

Actions are structured text strings in the format `action_type:key=value,...`

| Action | Task | Example |
|--------|------|---------|
| `fill_nulls` | Easy | `fill_nulls:mean_age=29.5,mean_salary=58750.0,ffill_city=Delhi` |
| `standardize` | Medium | `standardize:name=title_case,phone=digits_only,date=iso` |
| `clean_hard` | Hard | `clean_hard:remove_duplicates=true,replace_outliers=mean` |

The action space is intentionally **constrained and interpretable** — the agent must understand the data, compute the right values, and express a precise transformation. This makes the environment challenging for LLMs that reason poorly about numerical data.

---

## Observation Space

Each observation contains everything the agent needs:

| Field | Type | Description |
|-------|------|-------------|
| `task_name` | string | Which task is active (`easy`, `medium`, `hard`) |
| `description` | string | Plain-English instructions for the agent |
| `current_data` | string | The current (dirty or partially cleaned) dataset as CSV |
| `step_number` | int | How many steps have been taken |
| `previous_score` | float | Score after the last step (starts at 0.01) |

---

## Reward Function

The reward function provides **incremental feedback throughout the trajectory**, not just at completion:

```
reward = clamp(new_score - previous_score, 0.01, 0.99)
```

- Each step rewards the agent for measurable improvement
- A perfect clean in one step yields the maximum reward
- A bad action (parse error or regression) yields the minimum reward
- The episode ends when score reaches the threshold or max steps (5) is reached

This design encourages the agent to:
1. Compute correct values (not just guess)
2. Apply the right transformation type
3. Improve incrementally if the first attempt is imperfect

---

## Project Structure

```
├── inference.py          ← LLM agent (entry point for evaluation)
├── openenv.yaml          ← OpenEnv metadata and task registry
├── Dockerfile            ← HF Spaces deployment
├── requirements.txt      ← Python dependencies
├── README.md
├── env/
│   ├── __init__.py
│   ├── environment.py    ← DataCleaningEnv (reset, step, state, close)
│   ├── tasks.py          ← 3 task definitions with dirty + clean datasets
│   └── graders.py        ← Deterministic scoring logic (0.01–0.99)
└── server/
    └── app.py            ← FastAPI server via openenv-core
```

---

## Setup & Usage

### Local setup

```bash
git clone https://github.com/kartik10058/data-cleaning-env
cd data-cleaning-env
pip install -r requirements.txt

export HF_TOKEN=your_api_token
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini

python inference.py --task easy
```

### Docker

```bash
docker build -t data-cleaning-env .
docker run -e HF_TOKEN=<token> data-cleaning-env
```

### Expected Output

```
[START] task=easy env=data-cleaning-env model=gpt-4.1-mini
[STEP] step=1 action=fill_nulls:mean_age=29.5,mean_salary=58750.0,ffill_city=Delhi reward=0.98 done=true error=null
[END] success=true steps=1 rewards=0.98 score=0.98

[START] task=medium env=data-cleaning-env model=gpt-4.1-mini
[STEP] step=1 action=standardize:name=title_case,phone=digits_only,date=iso reward=0.98 done=true error=null
[END] success=true steps=1 rewards=0.98 score=0.98

[START] task=hard env=data-cleaning-env model=gpt-4.1-mini
[STEP] step=1 action=clean_hard:remove_duplicates=true,replace_outliers=mean reward=0.98 done=true error=null
[END] success=true steps=1 rewards=0.98 score=0.98
```

---

## Baseline Performance

| Task | Max Steps | Baseline Score | Notes |
|------|-----------|---------------|-------|
| Easy | 5 | 0.98 | Requires correct mean calculation |
| Medium | 5 | 0.98 | Requires correct format detection |
| Hard | 5 | 0.98 | Requires outlier detection + dedup |

---

## Design Decisions

**Why data cleaning?**
It's a universal, high-value real-world task with clear correctness criteria — making it ideal for RL evaluation. Unlike games or toy problems, every task maps to work humans actually do.

**Why structured action strings?**
They force the agent to reason explicitly about the data rather than outputting code or free text. This tests a specific capability: structured data reasoning under constrained output formats.

**Why incremental rewards?**
A sparse reward (only at episode end) makes learning harder and evaluation less informative. Incremental rewards let us observe partial progress and reward improvement at each step.

**Why 3 difficulty levels?**
Easy tests basic reasoning, Medium tests format understanding, Hard tests multi-problem detection. Together they form a meaningful benchmark with clear difficulty progression.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | ✅ Yes | — | API token for LLM calls |
| `API_BASE_URL` | No | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `gpt-4.1-mini` | Model identifier |
