import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import Environment, create_fastapi_app
from openenv.core.env_server.types import Action as BaseAction, State
from pydantic import BaseModel
from typing import Optional
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.tasks import get_task
from env.graders import score_progress

def clamp(v):
    return round(min(max(float(v), 0.001), 0.999), 4)

class DataAction(BaseAction):
    action_str: str = ""

class DataObservation(BaseModel):
    task_name: str = "easy"
    description: str = ""
    current_data: str = ""
    step_number: int = 0
    previous_score: float = 0.001
    reward: float = 0.001
    done: bool = False

class DataCleaningEnvironment(Environment):
    MAX_STEPS = 5

    def __init__(self):
        super().__init__()
        self.task_name = "easy"
        self.task_data = None
        self.current_df = None
        self.clean_df = None
        self.step_count = 0
        self.prev_score = 0.001
        self.done = False

    def reset(self) -> DataObservation:
        self.task_data = get_task(self.task_name)
        self.current_df = self.task_data["dirty_df"].copy()
        self.clean_df = self.task_data["clean_df"].copy()
        self.step_count = 0
        self.prev_score = 0.001
        self.done = False
        return DataObservation(
            task_name=self.task_name,
            description=self.task_data["description"],
            current_data=self.current_df.to_csv(index=False),
            step_number=0,
            previous_score=0.001,
            reward=0.001,
            done=False,
        )

    def step(self, action: DataAction) -> DataObservation:
        if self.done:
            return DataObservation(done=True, reward=0.001, previous_score=clamp(self.prev_score))

        self.step_count += 1
        try:
            self.current_df = self._apply_action(action.action_str)
        except Exception as e:
            done = self.step_count >= self.MAX_STEPS
            self.done = done
            return DataObservation(
                task_name=self.task_name,
                description=self.task_data["description"] if self.task_data else "",
                current_data=self.current_df.to_csv(index=False) if self.current_df is not None else "",
                step_number=self.step_count,
                previous_score=clamp(self.prev_score),
                reward=0.001,
                done=done,
            )

        new_score = score_progress(self.task_name, self.current_df, self.clean_df)
        reward = clamp(max(0.001, new_score - self.prev_score))
        self.done = (new_score >= 0.999) or (self.step_count >= self.MAX_STEPS)
        self.prev_score = new_score

        return DataObservation(
            task_name=self.task_name,
            description=self.task_data["description"],
            current_data=self.current_df.to_csv(index=False),
            step_number=self.step_count,
            previous_score=clamp(new_score),
            reward=reward,
            done=self.done,
        )

    @property
    def state(self):
        return {
            "task_name": self.task_name,
            "step_count": self.step_count,
            "prev_score": clamp(self.prev_score),
            "done": self.done,
        }

    def _apply_action(self, action_str: str) -> pd.DataFrame:
        df = self.current_df.copy()
        action_str = action_str.strip()
        if ":" not in action_str:
            raise ValueError(f"Invalid action format: {action_str}")
        action_type, params_str = action_str.split(":", 1)
        action_type = action_type.strip().lower()
        params = {}
        for part in params_str.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                params[k.strip()] = v.strip()
        if action_type == "fill_nulls":
            if "mean_age" in params:
                df["age"] = df["age"].fillna(float(params["mean_age"]))
            if "mean_salary" in params:
                df["salary"] = df["salary"].fillna(float(params["mean_salary"]))
            if "ffill_city" in params:
                df["city"] = df["city"].fillna(params["ffill_city"])
        elif action_type == "standardize":
            if params.get("name") == "title_case":
                df["name"] = df["name"].str.title()
            if params.get("phone") == "digits_only":
                df["phone"] = df["phone"].apply(lambda x: "".join(filter(str.isdigit, str(x))))
            if params.get("date") == "iso":
                df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True).dt.strftime("%Y-%m-%d")
        elif action_type == "clean_hard":
            if params.get("remove_duplicates") == "true":
                df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
            if params.get("replace_outliers") == "mean":
                for col in ["score", "salary"]:
                    col_data = df[col].astype(float)
                    mean = col_data.mean()
                    std = col_data.std()
                    is_outlier = (col_data - mean).abs() > 3 * std
                    clean_mean = col_data[~is_outlier].mean()
                    df[col] = col_data.where(~is_outlier, clean_mean)
        else:
            raise ValueError(f"Unknown action: {action_type}")
        return df

app = create_fastapi_app(DataCleaningEnvironment, DataAction, DataObservation)
