import pandas as pd
import numpy as np


def _clamp(score: float) -> float:
    """Map any score to strictly (0, 1) — never exactly 0.0 or 1.0."""
    score = max(0.0, min(1.0, score))
    return round(0.01 + score * 0.98, 4)


def grade(task_name: str, result_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    if task_name == "easy":
        return grade_easy(result_df, clean_df)
    elif task_name == "medium":
        return grade_medium(result_df, clean_df)
    elif task_name == "hard":
        return grade_hard(result_df, clean_df)
    else:
        raise ValueError(f"Unknown task: {task_name}")


def grade_easy(result_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    score = 0.0
    try:
        result_age = float(result_df.loc[3, "age"])
        clean_age  = float(clean_df.loc[3, "age"])
        if abs(result_age - clean_age) <= 0.5:
            score += 1

        result_salary = float(result_df.loc[3, "salary"])
        clean_salary  = float(clean_df.loc[3, "salary"])
        if abs(result_salary - clean_salary) <= 1.0:
            score += 1

        result_city = str(result_df.loc[3, "city"]).strip().lower()
        clean_city  = str(clean_df.loc[3, "city"]).strip().lower()
        if result_city == clean_city:
            score += 1
    except Exception:
        return _clamp(0.0)
    return _clamp(score / 3)


def grade_medium(result_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    columns_to_check = ["name", "phone", "date"]
    correct = 0
    total = 0
    try:
        for col in columns_to_check:
            for i in range(len(clean_df)):
                total += 1
                result_val = str(result_df.loc[i, col]).strip()
                clean_val  = str(clean_df.loc[i, col]).strip()
                if col == "name":
                    if result_val.lower() == clean_val.lower():
                        correct += 1
                elif col == "phone":
                    if "".join(filter(str.isdigit, result_val)) == "".join(filter(str.isdigit, clean_val)):
                        correct += 1
                elif col == "date":
                    if result_val == clean_val:
                        correct += 1
    except Exception:
        return _clamp(0.0)
    return _clamp(correct / total if total > 0 else 0.0)


def grade_hard(result_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    dedup_score   = 0.0
    outlier_score = 0.0
    try:
        if len(result_df) == len(clean_df):
            dedup_score += 0.25
        if sorted(result_df["id"].tolist()) == sorted(clean_df["id"].tolist()):
            dedup_score += 0.25

        david_row = result_df[result_df["id"] == 4]
        if not david_row.empty:
            if abs(float(david_row["score"].values[0]) - float(clean_df[clean_df["id"] == 4]["score"].values[0])) <= 1.0:
                outlier_score += 0.25

        eva_row = result_df[result_df["id"] == 5]
        if not eva_row.empty:
            if abs(float(eva_row["salary"].values[0]) - float(clean_df[clean_df["id"] == 5]["salary"].values[0])) <= 100.0:
                outlier_score += 0.25
    except Exception:
        return _clamp(0.0)
    return _clamp(dedup_score + outlier_score)


def score_progress(task_name: str, result_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    return grade(task_name, result_df, clean_df)
