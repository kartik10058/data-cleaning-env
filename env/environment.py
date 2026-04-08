import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Any, Optional
from env.tasks import get_task
from env.graders import grade, score_progress


# ─────────────────────────────────────────────
# PYDANTIC MODELS
# These define the "shape" of data flowing in/out
# OpenEnv requires typed models for Observation, Action, Reward
# ─────────────────────────────────────────────

class Observation(BaseModel):
    task_name: str                  # "easy", "medium", or "hard"
    description: str                # instructions for the LLM
    current_data: str               # the dirty dataframe as a CSV string
    step_number: int                # how many steps taken so far
    previous_score: float           # score from last step (0.0 if first step)

class Action(BaseModel):
    action_str: str                 # the raw action string from the LLM
                                    # e.g. "fill_nulls:mean_age=29.5,mean_salary=58750.0,ffill_city=Delhi"

class Reward(BaseModel):
    value: float                    # between 0.0 and 1.0
    done: bool                      # True if task is complete or max steps reached
    info: dict                      # extra debug info


# ─────────────────────────────────────────────
# THE ENVIRONMENT CLASS
# ─────────────────────────────────────────────

class DataCleaningEnv:
    """
    OpenEnv-compatible environment for data cleaning tasks.
    Supports three tasks: easy, medium, hard.
    """

    MAX_STEPS = 5   # agent gets at most 5 attempts per episode

    def __init__(self, task_name: str = "easy"):
        self.task_name    = task_name
        self.task_data    = None    # loaded on reset()
        self.current_df   = None   # the working (possibly partially cleaned) dataframe
        self.clean_df     = None   # the ground truth answer
        self.step_count   = 0
        self.prev_score   = 0.0
        self.last_error   = None   # last action error message, or None
        self.done         = False

    # ── reset() ──────────────────────────────
    def reset(self) -> Observation:
        """
        Starts a new episode.
        Loads the task, resets state, returns the first observation.
        """
        self.task_data  = get_task(self.task_name)
        self.current_df = self.task_data["dirty_df"].copy()
        self.clean_df   = self.task_data["clean_df"].copy()
        self.step_count = 0
        self.prev_score = 0.0
        self.last_error = None
        self.done       = False

        return self._make_observation()

    # ── step() ───────────────────────────────
    def step(self, action: Action):
        """
        Agent submits an action string.
        Environment applies it, scores the result, returns reward.

        Returns: (Observation, Reward)
        """
        if self.done:
            # Episode already over — return zero reward
            return self._make_observation(), Reward(value=0.0, done=True, info={"error": "Episode already done"})

        self.step_count += 1
        self.last_error = None

        # ── Apply the action ──
        try:
            self.current_df = self._apply_action(action.action_str)
        except Exception as e:
            self.last_error = str(e)
            # Bad action — no change to dataframe, small penalty
            reward_value = max(0.0, self.prev_score - 0.1)
            done = self.step_count >= self.MAX_STEPS
            self.done = done
            return self._make_observation(), Reward(
                value=round(reward_value, 2),
                done=done,
                info={"error": self.last_error, "step": self.step_count}
            )

        # ── Score the result ──
        new_score = score_progress(self.task_name, self.current_df, self.clean_df)

        # Incremental reward = improvement since last step
        incremental = new_score - self.prev_score
        reward_value = max(0.0, round(incremental, 2))

        # Episode is done if perfect score OR max steps reached
        self.done = (new_score >= 1.0) or (self.step_count >= self.MAX_STEPS)
        self.prev_score = new_score

        return self._make_observation(), Reward(
            value=reward_value,
            done=self.done,
            info={"score": new_score, "step": self.step_count}
        )

    # ── state() ──────────────────────────────
    def state(self) -> dict:
        """
        Returns current internal state.
        Required by OpenEnv spec.
        """
        return {
            "task_name":   self.task_name,
            "step_count":  self.step_count,
            "prev_score":  self.prev_score,
            "done":        self.done,
            "last_error":  self.last_error,
            "current_data": self.current_df.to_csv(index=False) if self.current_df is not None else "",
        }

    # ── close() ──────────────────────────────
    def close(self):
        """Cleanup. Required by OpenEnv spec."""
        self.current_df = None
        self.clean_df   = None
        self.done       = True

    # ─────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────

    def _make_observation(self) -> Observation:
        """Packages current state into an Observation object."""
        return Observation(
            task_name     = self.task_name,
            description   = self.task_data["description"] if self.task_data else "",
            current_data  = self.current_df.to_csv(index=False) if self.current_df is not None else "",
            step_number   = self.step_count,
            previous_score= self.prev_score,
        )

    def _apply_action(self, action_str: str) -> pd.DataFrame:
        """
        Parses the action string and applies the transformation to current_df.
        Returns the modified dataframe.

        Supported actions:
          fill_nulls:mean_age=X,mean_salary=Y,ffill_city=Z   (easy task)
          standardize:name=title_case,phone=digits_only,date=iso  (medium task)
          clean_hard:remove_duplicates=true,replace_outliers=mean  (hard task)
        """
        df = self.current_df.copy()
        action_str = action_str.strip()

        # ── Parse action type and params ──
        if ":" not in action_str:
            raise ValueError(f"Invalid action format. Expected 'action_type:params', got: '{action_str}'")

        action_type, params_str = action_str.split(":", 1)
        action_type = action_type.strip().lower()

        # Parse key=value pairs
        params = {}
        for part in params_str.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                params[k.strip()] = v.strip()

        # ── EASY: fill nulls ──
        if action_type == "fill_nulls":
            if "mean_age" in params:
                df["age"] = df["age"].fillna(float(params["mean_age"]))
            if "mean_salary" in params:
                df["salary"] = df["salary"].fillna(float(params["mean_salary"]))
            if "ffill_city" in params:
                df["city"] = df["city"].fillna(params["ffill_city"])

        # ── MEDIUM: standardize formats ──
        elif action_type == "standardize":
            if params.get("name") == "title_case":
                df["name"] = df["name"].str.title()
            if params.get("phone") == "digits_only":
                df["phone"] = df["phone"].apply(
                    lambda x: "".join(filter(str.isdigit, str(x)))
                )
            if params.get("date") == "iso":
                df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True).dt.strftime("%Y-%m-%d")

        # ── HARD: dedup + outlier removal ──
        elif action_type == "clean_hard":
            if params.get("remove_duplicates") == "true":
                df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
            if params.get("replace_outliers") == "mean":
                for col in ["score", "salary"]:
                    col_data = df[col].astype(float)
                    mean = col_data.mean()
                    std  = col_data.std()
                    # Outlier = more than 3 std deviations from mean
                    is_outlier = (col_data - mean).abs() > 3 * std
                    # Replace outliers with mean of non-outlier values
                    clean_mean = col_data[~is_outlier].mean()
                    df[col] = col_data.where(~is_outlier, clean_mean)

        else:
            raise ValueError(f"Unknown action type: '{action_type}'. Use fill_nulls, standardize, or clean_hard.")

        return df
