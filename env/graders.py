import pandas as pd
import numpy as np


def _clamp(score: float) -> float:
    return round(min(max(score, 0.001), 0.999), 4)


def grade(task_name: str, result_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    if task_name == "easy":
        return _clamp(grade_easy(result_df, clean_df))
    elif task_name == "medium":
        return _clamp(grade_medium(result_df, clean_df))
    elif task_name == "hard":
        return _clamp(grade_hard(result_df, clean_df))
    else:
        raise ValueError(f"Unknown task: {task_name}")


def grade_easy(result_df, clean_df):
    score = 0.0
    try:
        if abs(float(result_df.loc[3,"age"]) - float(clean_df.loc[3,"age"])) <= 0.5:
            score += 1
        if abs(float(result_df.loc[3,"salary"]) - float(clean_df.loc[3,"salary"])) <= 1.0:
            score += 1
        if str(result_df.loc[3,"city"]).strip().lower() == str(clean_df.loc[3,"city"]).strip().lower():
            score += 1
    except:
        return 0.001
    return score / 3


def grade_medium(result_df, clean_df):
    correct, total = 0, 0
    try:
        for col in ["name","phone","date"]:
            for i in range(len(clean_df)):
                total += 1
                r = str(result_df.loc[i,col]).strip()
                c = str(clean_df.loc[i,col]).strip()
                if col == "name" and r.lower() == c.lower(): correct += 1
                elif col == "phone" and "".join(filter(str.isdigit,r)) == "".join(filter(str.isdigit,c)): correct += 1
                elif col == "date" and r == c: correct += 1
    except:
        return 0.001
    return correct / total if total > 0 else 0.001


def grade_hard(result_df, clean_df):
    d, o = 0.0, 0.0
    try:
        if len(result_df) == len(clean_df): d += 0.25
        if sorted(result_df["id"].tolist()) == sorted(clean_df["id"].tolist()): d += 0.25
        dr = result_df[result_df["id"]==4]
        if not dr.empty and abs(float(dr["score"].values[0]) - float(clean_df[clean_df["id"]==4]["score"].values[0])) <= 1.0: o += 0.25
        er = result_df[result_df["id"]==5]
        if not er.empty and abs(float(er["salary"].values[0]) - float(clean_df[clean_df["id"]==5]["salary"].values[0])) <= 100.0: o += 0.25
    except:
        return 0.001
    return d + o


def score_progress(task_name, result_df, clean_df):
    return _clamp(grade(task_name, result_df, clean_df))
