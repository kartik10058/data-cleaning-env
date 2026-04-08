"""
app.py — Web server for HF Spaces.

Exposes both:
  1. A UI at GET / for running inference manually
  2. OpenEnv API endpoints the validator calls:
       POST /reset
       POST /step
       GET  /state
       GET  /health
"""

import os
import sys
import subprocess
from flask import Flask, Response, request, jsonify

sys.path.insert(0, os.path.dirname(__file__))
from env.environment import DataCleaningEnv, Action

app = Flask(__name__)

# Global environment instance (one per server process)
_env: DataCleaningEnv = None
_current_task = "easy"


def get_env() -> DataCleaningEnv:
    global _env
    if _env is None:
        _env = DataCleaningEnv(task_name=_current_task)
    return _env


# ── HTML template ─────────────────────────────────────────────────────────────
PAGE = """<!DOCTYPE html>
<html>
<head>
  <title>Data Cleaning OpenEnv</title>
  <style>
    body {{ font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 2rem; }}
    h1   {{ color: #58a6ff; }}
    pre  {{ background: #161b22; padding: 1rem; border-radius: 6px; white-space: pre-wrap; }}
    a    {{ color: #58a6ff; margin-right: 1rem; }}
  </style>
</head>
<body>
  <h1>🧹 Data Cleaning OpenEnv</h1>
  <p>An OpenEnv RL environment for data cleaning tasks.</p>
  <p>
    <a href="/run?task=easy">▶ Run Easy Task</a>
    <a href="/run?task=medium">▶ Run Medium Task</a>
    <a href="/run?task=hard">▶ Run Hard Task</a>
    <a href="/run_all">▶ Run All Tasks</a>
  </p>
  <pre>{output}</pre>
</body>
</html>"""

STATUS_MSG = (
    "Server is running.\n\n"
    "Click one of the links above to run inference.\n\n"
    "Tasks:\n"
    "  easy   — fill missing values\n"
    "  medium — standardize formats\n"
    "  hard   — remove duplicates + outliers\n"
)


# ── UI Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return PAGE.format(output=STATUS_MSG)


@app.route("/run")
def run_task():
    task = request.args.get("task", "easy")
    if task not in ("easy", "medium", "hard"):
        return PAGE.format(output=f"Unknown task: {task}. Use easy, medium, or hard."), 400
    output = run_inference(task)
    return PAGE.format(output=output)


@app.route("/run_all")
def run_all():
    results = []
    for task in ["easy", "medium", "hard"]:
        results.append(f"{'='*50}\nTASK: {task}\n{'='*50}")
        results.append(run_inference(task))
    return PAGE.format(output="\n\n".join(results))


# ── OpenEnv API Routes ────────────────────────────────────────────────────────

@app.route("/reset", methods=["POST"])
def reset():
    """
    OpenEnv validator calls POST /reset to start a new episode.
    Body (optional JSON): {"task": "easy"|"medium"|"hard"}
    Returns the initial observation as JSON.
    """
    global _env, _current_task

    data = request.get_json(silent=True) or {}
    task = data.get("task", "easy")
    if task not in ("easy", "medium", "hard"):
        task = "easy"

    _current_task = task
    _env = DataCleaningEnv(task_name=task)
    obs = _env.reset()

    return jsonify({
        "task_name":      obs.task_name,
        "description":    obs.description,
        "current_data":   obs.current_data,
        "step_number":    obs.step_number,
        "previous_score": obs.previous_score,
    })


@app.route("/step", methods=["POST"])
def step():
    """
    OpenEnv validator calls POST /step to take an action.
    Body JSON: {"action": "<action_string>"}
    Returns observation + reward as JSON.
    """
    env = get_env()
    data = request.get_json(silent=True) or {}
    action_str = data.get("action", "")

    obs, reward = env.step(Action(action_str=action_str))

    return jsonify({
        "observation": {
            "task_name":      obs.task_name,
            "description":    obs.description,
            "current_data":   obs.current_data,
            "step_number":    obs.step_number,
            "previous_score": obs.previous_score,
        },
        "reward": reward.value,
        "done":   reward.done,
        "info":   reward.info,
    })


@app.route("/state", methods=["GET"])
def state():
    """
    OpenEnv validator calls GET /state to check current state.
    """
    env = get_env()
    return jsonify(env.state())


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ── Inference runner (for UI) ─────────────────────────────────────────────────

def run_inference(task_name: str) -> str:
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "inference.py", "--task", task_name],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    output = result.stdout
    if result.stderr:
        output += f"\n\n--- stderr ---\n{result.stderr}"
    return output or "(no output)"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting server on port {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
