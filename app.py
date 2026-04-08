"""
app.py — Web server wrapper for HF Spaces.

HF Spaces requires a long-running HTTP server on port 7860.
This file starts a Flask server that:
  - Shows a status page at GET /
  - Runs inference for a task at GET /run?task=easy|medium|hard
  - Runs all tasks at GET /run_all
"""

import os
import sys
import threading
import subprocess
from flask import Flask, Response, request

app = Flask(__name__)

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
    .tag {{ color: #3fb950; font-weight: bold; }}
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

# ── Routes ────────────────────────────────────────────────────────────────────

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


@app.route("/health")
def health():
    return {"status": "ok"}, 200


# ── Inference runner ──────────────────────────────────────────────────────────

def run_inference(task_name: str) -> str:
    """
    Runs inference.py for one task as a subprocess.
    Captures stdout + stderr and returns it as a string.
    This avoids import conflicts and gives clean isolated output.
    """
    env = os.environ.copy()
    env["SINGLE_TASK"] = task_name   # inference.py reads this if set

    result = subprocess.run(
        [sys.executable, "inference.py", "--task", task_name],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,   # 2 min max per task
    )

    output = result.stdout
    if result.stderr:
        output += f"\n\n--- stderr ---\n{result.stderr}"
    if not output.strip():
        output = "(no output)"
    return output


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting server on port {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
