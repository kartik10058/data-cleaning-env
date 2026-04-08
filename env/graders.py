import pandas as pd
import numpy as np


def grade(task_name: str, result_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """
    Master grader — routes to the correct grader based on task name.
    Always returns a float between 0.0 and 1.0.
    """
    if task_name == "easy":
        return grade_easy(result_df, clean_df)
    elif task_name == "medium":
        return grade_medium(result_df, clean_df)
    elif task_name == "hard":
        return grade_hard(result_df, clean_df)
    else:
        raise ValueError(f"Unknown task: {task_name}")


# ─────────────────────────────────────────────
# GRADER 1 — EASY: Score null filling
# ─────────────────────────────────────────────
def grade_easy(result_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """
    Checks 3 specific cells that were originally null:
      - age of David (row 3)
      - salary of David (row 3)
      - city of David (row 3)

    Each correct cell = 1/3 of the score.
    We allow a small tolerance (±0.5) for numeric values.
    """
    score = 0.0
    total_checks = 3

    try:
        # Check age (numeric — allow small rounding difference)
        result_age = float(result_df.loc[3, "age"])
        clean_age  = float(clean_df.loc[3, "age"])
        if abs(result_age - clean_age) <= 0.5:
            score += 1

        # Check salary (numeric)
        result_salary = float(result_df.loc[3, "salary"])
        clean_salary  = float(clean_df.loc[3, "salary"])
        if abs(result_salary - clean_salary) <= 1.0:
            score += 1

        # Check city (string — exact match, case-insensitive)
        result_city = str(result_df.loc[3, "city"]).strip().lower()
        clean_city  = str(clean_df.loc[3, "city"]).strip().lower()
        if result_city == clean_city:
            score += 1

    except Exception:
        # If anything goes wrong (wrong shape, missing column), score 0
        return 0.0

    return round(score / total_checks, 2)


# ─────────────────────────────────────────────
# GRADER 2 — MEDIUM: Score format standardization
# ─────────────────────────────────────────────
def grade_medium(result_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """
    Checks every cell in the name, phone, and date columns.
    Score = (number of correct cells) / (total cells to check)
    """
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
                    # Title case comparison
                    if result_val.lower() == clean_val.lower():
                        correct += 1
                elif col == "phone":
                    # Strip all non-digits before comparing
                    result_digits = "".join(filter(str.isdigit, result_val))
                    clean_digits  = "".join(filter(str.isdigit, clean_val))
                    if result_digits == clean_digits:
                        correct += 1
                elif col == "date":
                    # Exact string match for ISO format
                    if result_val == clean_val:
                        correct += 1

    except Exception:
        return 0.0

    return round(correct / total, 2) if total > 0 else 0.0


# ─────────────────────────────────────────────
# GRADER 3 — HARD: Score dedup + outlier removal
# ─────────────────────────────────────────────
def grade_hard(result_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """
    Two sub-scores, equally weighted:
      1. Deduplication score  (0.0 to 0.5)
         — Did the agent remove the duplicate row?
         — Is the row count correct?
      2. Outlier replacement score (0.0 to 0.5)
         — Were outlier values replaced with correct means?

    Total = dedup_score + outlier_score (max 1.0)
    """
    dedup_score   = 0.0
    outlier_score = 0.0

    try:
        # ── Sub-score 1: Deduplication ──
        # Expected: 5 rows, unique ids [1,2,3,4,5]
        if len(result_df) == len(clean_df):
            dedup_score += 0.25  # correct row count

        result_ids = sorted(result_df["id"].tolist())
        clean_ids  = sorted(clean_df["id"].tolist())
        if result_ids == clean_ids:
            dedup_score += 0.25  # correct unique ids

        # ── Sub-score 2: Outlier replacement ──
        # Find rows where clean_df differs from dirty baseline
        # Check score for David (id=4) and salary for Eva (id=5)

        # Score outlier — David's score should be ~85.25
        david_row = result_df[result_df["id"] == 4]
        if not david_row.empty:
            result_score = float(david_row["score"].values[0])
            clean_score  = float(clean_df[clean_df["id"] == 4]["score"].values[0])
            if abs(result_score - clean_score) <= 1.0:
                outlier_score += 0.25

        # Salary outlier — Eva's salary should be ~58750
        eva_row = result_df[result_df["id"] == 5]
        if not eva_row.empty:
            result_sal = float(eva_row["salary"].values[0])
            clean_sal  = float(clean_df[clean_df["id"] == 5]["salary"].values[0])
            if abs(result_sal - clean_sal) <= 100.0:
                outlier_score += 0.25

    except Exception:
        return 0.0

    total = round(dedup_score + outlier_score, 2)
    return total


# ─────────────────────────────────────────────
# UTILITY: Partial progress scorer
# ─────────────────────────────────────────────
def score_progress(task_name: str, result_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """
    Same as grade() but used mid-episode to give incremental rewards.
    The environment calls this after every step to measure improvement.
    """
    return grade(task_name, result_df, clean_df)
