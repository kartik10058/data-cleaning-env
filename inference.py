import os
import sys
import argparse
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from env.environment import DataCleaningEnv, Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required but not set.")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def clamp(v):
    return round(min(max(float(v), 0.01), 0.99), 4)

SYSTEM_PROMPT = """You are a data cleaning agent. Respond with EXACTLY ONE action string, nothing else.

1. EASY task: fill_nulls:mean_age=<number>,mean_salary=<number>,ffill_city=<city>
2. MEDIUM task: standardize:name=title_case,phone=digits_only,date=iso
3. HARD task: clean_hard:remove_duplicates=true,replace_outliers=mean"""

def ask_llm(description, current_data, previous_score, step):
    msg = f"TASK:\n{description}\n\nDATA (CSV):\n{current_data}\n\nSCORE: {previous_score}\nSTEP: {step}\n\nRespond with ONLY the action string."
    r = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":msg}],
        max_tokens=200, temperature=0.0,
    )
    return r.choices[0].message.content.strip()

def run_episode(task_name):
    env = DataCleaningEnv(task_name=task_name)
    print(f"[START] task={task_name} env=data-cleaning-env model={MODEL_NAME}", flush=True)
    rewards_log, success, final_steps = [], False, 0

    try:
        obs = env.reset()
        while True:
            try:
                action_str = ask_llm(obs.description, obs.current_data, obs.previous_score, obs.step_number+1)
            except Exception as e:
                action_str = "fill_nulls:mean_age=29.5,mean_salary=58750.0,ffill_city=Delhi"
                r = clamp(0.001)
                print(f"[STEP] step={obs.step_number+1} action={action_str} reward={r:.4f} done=false error=LLM_ERROR:{str(e)[:80]}", flush=True)
                rewards_log.append(r)
                break

            obs, reward = env.step(Action(action_str=action_str))
            r = clamp(reward.value)
            done_str = "true" if reward.done else "false"
            error_str = (reward.info.get("error") or "null")
            if error_str != "null":
                error_str = error_str.replace("\n"," ")[:120]

            print(f"[STEP] step={obs.step_number} action={action_str} reward={r:.4f} done={done_str} error={error_str}", flush=True)
            rewards_log.append(r)

            if reward.done:
                final_score = clamp(reward.info.get("score", obs.previous_score))
                success = final_score >= 0.99
                final_steps = obs.step_number
                break

    except Exception as e:
        final_steps = final_steps or 1
        r = clamp(0.001)
        print(f"[STEP] step={final_steps} action=error reward={r:.4f} done=true error={str(e)[:120]}", flush=True)
        rewards_log.append(r)
    finally:
        env.close()

    rewards_str = ",".join(f"{r:.4f}" for r in rewards_log) if rewards_log else "0.001"
   avg_score = round(min(max(sum(rewards_log) / len(rewards_log) if rewards_log else 0.01, 0.01), 0.99), 2)
print(f"[END] success={'true' if success else 'false'} steps={final_steps} rewards={rewards_str} score={avg_score}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["easy","medium","hard","all"], default="all")
    args = parser.parse_args()
    tasks = ["easy","medium","hard"] if args.task == "all" else [args.task]
    for task in tasks:
        run_episode(task_name=task)
        print()
