"""
inference.py — LLM agent for the DataCleaning OpenEnv environment.

Reads environment variables:
  API_BASE_URL  — LLM API endpoint  (has default)
  MODEL_NAME    — model identifier   (has default)
  HF_TOKEN      — Hugging Face token (required, no default)

Emits exactly three line types to stdout:
  [START] task=<n> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import os
import sys
import argparse

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from env.environment import DataCleaningEnv, Action

# ── Environment variables ─────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required but not set.")

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a data cleaning agent operating inside an RL environment.

You will receive:
- A description of the data cleaning task
- The current state of the dataset (as CSV text)
- Your previous score (0.0 to 1.0)

Your job is to respond with EXACTLY ONE action string — nothing else.
No explanation, no markdown, no extra text. Just the action string.

The three possible action formats are:

1. For EASY task (fill missing values):
   fill_nulls:mean_age=<number>,mean_salary=<number>,ffill_city=<city_name>
   Example: fill_nulls:mean_age=29.5,mean_salary=58750.0,ffill_city=Delhi

2. For MEDIUM task (fix formats):
   standardize:name=title_case,phone=digits_only,date=iso

3. For HARD task (remove duplicates and outliers):
   clean_hard:remove_duplicates=true,replace_outliers=mean

Read the task description carefully to know which action to use.
For the easy task, calculate the mean from the non-null values in the CSV.
"""

# ── Ask the LLM ───────────────────────────────────────────────────────────────
def ask_llm(task_description: str, current_data: str, previous_score: float, step: int) -> str:
    user_message = f"""TASK DESCRIPTION:
{task_description}

CURRENT DATA (CSV):
{current_data}

YOUR PREVIOUS SCORE: {previous_score:.2f}
STEP NUMBER: {step}

Respond with ONLY the action string. Nothing else."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=200,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

# ── Run one episode ───────────────────────────────────────────────────────────
def run_episode(task_name: str) -> None:
    env = DataCleaningEnv(task_name=task_name)
    env_name = "data-cleaning-env"

    print(f"[START] task={task_name} env={env_name} model={MODEL_NAME}", flush=True)

    rewards_log = []
    success     = False
    final_steps = 0

    try:
        obs = env.reset()

        while True:
            try:
                action_str = ask_llm(
                    task_description = obs.description,
                    current_data     = obs.current_data,
                    previous_score   = obs.previous_score,
                    step             = obs.step_number + 1,
                )
            except Exception as llm_err:
                action_str = "fill_nulls:mean_age=0,mean_salary=0,ffill_city=unknown"
                print(
                    f"[STEP] step={obs.step_number + 1} action={action_str} "
                    f"reward=0.00 done=false error=LLM_ERROR:{str(llm_err)[:80]}",
                    flush=True
                )
                rewards_log.append(0.0)
                break

            obs, reward = env.step(Action(action_str=action_str))

            done_str  = "true" if reward.done else "false"
            error_str = reward.info.get("error") or "null"
            if error_str != "null":
                error_str = error_str.replace("\n", " ")[:120]

            print(
                f"[STEP] step={obs.step_number} "
                f"action={action_str} "
                f"reward={reward.value:.2f} "
                f"done={done_str} "
                f"error={error_str}",
                flush=True
            )

            rewards_log.append(reward.value)

            if reward.done:
                final_score = reward.info.get("score", obs.previous_score)
                success     = final_score >= 1.0
                final_steps = obs.step_number
                break

    except Exception as e:
        final_steps = final_steps or 1
        error_msg = str(e).replace("\n", " ")[:120]
        print(f"[STEP] step={final_steps} action=error reward=0.00 done=true error={error_msg}", flush=True)
        rewards_log.append(0.0)

    finally:
        env.close()

    rewards_str = ",".join(f"{r:.2f}" for r in rewards_log) if rewards_log else "0.00"
    success_str = "true" if success else "false"
    print(
        f"[END] success={success_str} steps={final_steps} rewards={rewards_str}",
        flush=True
    )

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    for task in tasks:
        run_episode(task_name=task)
        print()
