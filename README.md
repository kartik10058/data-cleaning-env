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

An OpenEnv-compatible reinforcement learning environment where an LLM agent cleans messy CSV datasets across three difficulty levels.

---

## Overview

Real-world data is messy. This environment simulates common data quality problems that analysts and data engineers face every day:

| Task | Problem | Difficulty |
|------|---------|------------|
| Easy | Missing values (nulls) | ⭐ Easy |
| Medium | Inconsistent formats (casing, phones, dates) | ⭐⭐ Medium |
| Hard | Duplicate rows + numeric outliers | ⭐⭐⭐ Hard |

The agent reads a dirty dataset as CSV text, decides what cleaning action to apply, and receives a reward proportional to how much it improved the data.

---

## Action Space

Actions are structured text strings in the format `action_type:key=value,...`

| Action | When to use | Example |
|--------|------------|---------|
| `fill_nulls` | Easy task — fill NaN values | `fill_nulls:mean_age=29.5,mean_salary=58750.0,ffill_city=Delhi` |
| `standardize` | Medium task — fix formats | `standardize:name=title_case,phone=digits_only,date=iso` |
| `clean_hard` | Hard task — dedup + outliers | `clean_hard:remove_duplicates=true,replace_outliers=mean` |

---

## Observation Space

Each observation is a JSON-serialisable object with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `task_name` | string | `"easy"`, `"medium"`, or `"hard"` |
| `description` | string | Plain-English instructions for the agent |
| `current_data` | string | The current (dirty or partially cleaned) dataframe as CSV text |
| `step_number` | int | How many steps have been taken so far |
| `previous_score` | float | The agent's score after the last step (0.0 on first step) |

---

## Reward Function

- **Incremental reward** = new score − previous score after each step
- Score is always between 0.0 and 1.0 (percentage of cells correctly cleaned)
- A perfect clean in one step yields `reward = 1.0, done = true`
- A bad action (parse error, or regression) yields `reward = 0.0` and a small score penalty

This satisfies the OpenEnv requirement of rewarding **progress throughout the trajectory**, not only at completion.

---

## Task Descriptions

### Easy — Fill Missing Values
The dataset has 3 null cells (age, salary, city for one row). The agent must fill them using:
- Numeric columns → mean of the non-null values
- Text columns → forward-fill (copy from the row above)

**Baseline score:** 1.0 (solvable in a single step if means are computed correctly)

### Medium — Standardize Formats
The dataset has inconsistent formatting across 3 columns:
- `name` — mixed casing → fix to Title Case
- `phone` — various formats (dashes, brackets, spaces) → 10 digits only
- `date` — various date formats → ISO 8601 (`YYYY-MM-DD`)

**Baseline score:** 1.0 (solvable in a single step with `standardize:name=title_case,phone=digits_only,date=iso`)

### Hard — Duplicates + Outliers
The dataset has two problems:
- One duplicate row (same `id`, keep the first occurrence)
- Two outlier values (score=999 and salary=9999999) to replace with column means

**Baseline score:** 1.0 (solvable in a single step with `clean_hard:remove_duplicates=true,replace_outliers=mean`)

---

## Project Structure

```
├── inference.py          ← LLM agent (entry point)
├── openenv.yaml          ← OpenEnv metadata
├── Dockerfile            ← HF Spaces deployment
├── requirements.txt
├── README.md
└── env/
    ├── environment.py    ← DataCleaningEnv (step, reset, state, close)
    ├── tasks.py          ← 3 task definitions (dirty + clean datasets)
    └── graders.py        ← Scoring logic (0.0–1.0 per task)
```

---

## Setup & Usage

### Local setup

```bash
# Clone / copy the project
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN=your_huggingface_token
export API_BASE_URL=https://api.openai.com/v1   # or your custom endpoint
export MODEL_NAME=gpt-4.1-mini

# Run
python inference.py
```

### Docker

```bash
docker build -t data-cleaning-env .
docker run -e HF_TOKEN=<your_token> data-cleaning-env
```

### Expected output

```
[START] task=easy env=data-cleaning-env model=gpt-4.1-mini
[STEP] step=1 action=fill_nulls:mean_age=29.5,mean_salary=58750.0,ffill_city=Delhi reward=1.00 done=true error=null
[END] success=true steps=1 rewards=1.00

[START] task=medium env=data-cleaning-env model=gpt-4.1-mini
[STEP] step=1 action=standardize:name=title_case,phone=digits_only,date=iso reward=1.00 done=true error=null
[END] success=true steps=1 rewards=1.00

[START] task=hard env=data-cleaning-env model=gpt-4.1-mini
[STEP] step=1 action=clean_hard:remove_duplicates=true,replace_outliers=mean reward=1.00 done=true error=null
[END] success=true steps=1 rewards=1.00
```

---

## Baseline Performance Scores

| Task | Max Steps | Best Possible Score | Notes |
|------|-----------|-------------------|-------|
| Easy | 5 | 1.0 | Solvable in 1 step |
| Medium | 5 | 1.0 | Solvable in 1 step |
| Hard | 5 | 1.0 | Solvable in 1 step |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | ✅ Yes | — | Hugging Face / LLM API token |
| `API_BASE_URL` | No | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `gpt-4.1-mini` | Model to use for inference |
