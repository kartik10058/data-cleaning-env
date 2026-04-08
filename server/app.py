"""
server/app.py — OpenEnv-compatible REST API server.
Required by openenv validate for multi-mode deployment.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, request, jsonify
from env.environment import DataCleaningEnv, Action

app = Flask(__name__)

# Global env instance
_env: DataCleaningEnv = None


def get_env() -> DataCleaningEnv:
    global _env
    if _env is None:
        _env = DataCleaningEnv(task_name="easy")
        _env.reset()
    return _env


@app.route("/reset", methods=["POST"])
def reset():
    global _env
    data = request.get_json(silent=True) or {}
    task = data.get("task", data.get("task_name", "easy"))
    if task not in ("easy", "medium", "hard"):
        task = "easy"
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
    return jsonify(get_env().state())


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name": "data-cleaning-env",
        "version": "1.0.0",
        "tasks": ["easy", "medium", "hard"],
    })


def main():
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting OpenEnv server on port {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
