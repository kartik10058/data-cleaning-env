import pandas as pd
import numpy as np

def get_task(task_name: str):
    """
    Returns a dict with:
    - dirty_df   : the messy dataframe the agent will try to clean
    - clean_df   : the correct answer (used by grader)
    - description: text the LLM reads to understand what to do
    """
    if task_name == "easy":
        return easy_task()
    elif task_name == "medium":
        return medium_task()
    elif task_name == "hard":
        return hard_task()
    else:
        raise ValueError(f"Unknown task: {task_name}")


# ─────────────────────────────────────────────
# TASK 1 — EASY: Fill missing values
# ─────────────────────────────────────────────
def easy_task():
    # The clean version (ground truth)
    clean_df = pd.DataFrame({
        "name":   ["Alice", "Bob", "Carol", "David", "Eva"],
        "age":    [25, 30, 35, 30, 28],        # mean of [25,30,35,28] = 29.5 → rounded = 30
        "salary": [50000, 60000, 70000, 60000, 55000],  # mean fill
        "city":   ["Chennai", "Mumbai", "Delhi", "Chennai", "Mumbai"],  # forward fill
    })

    # The dirty version (what agent sees)
    dirty_df = pd.DataFrame({
        "name":   ["Alice", "Bob", "Carol", "David", "Eva"],
        "age":    [25, 30, 35, None, 28],       # 1 missing — fill with mean (30)
        "salary": [50000, 60000, 70000, None, 55000],  # 1 missing — fill with mean (58750 → use 60000 clean)
        "city":   ["Chennai", "Mumbai", "Delhi", None, "Mumbai"],  # 1 missing — forward fill → "Delhi"
    })

    # Fix clean_df to match exact expected values
    clean_df = pd.DataFrame({
        "name":   ["Alice", "Bob", "Carol", "David", "Eva"],
        "age":    [25.0, 30.0, 35.0, 29.5, 28.0],
        "salary": [50000.0, 60000.0, 70000.0, 58750.0, 55000.0],
        "city":   ["Chennai", "Mumbai", "Delhi", "Delhi", "Mumbai"],
    })

    description = """
You are a data cleaning agent. You have a CSV dataset with MISSING VALUES (shown as null/NaN).

Your job:
1. Fill missing numeric values (age, salary) with the MEAN of that column.
2. Fill missing text values (city) with FORWARD FILL (copy the value from the row above).

The dataset has these columns: name, age, salary, city.

To submit your answer, call the action: fill_nulls
Your action string must look like this:
  fill_nulls:mean_age=<value>,mean_salary=<value>,ffill_city=<value>

Example:
  fill_nulls:mean_age=29.5,mean_salary=58750.0,ffill_city=Delhi

Calculate the correct mean values from the non-null entries and fill in the action string.
"""
    return {
        "dirty_df": dirty_df,
        "clean_df": clean_df,
        "description": description,
        "task_name": "easy",
    }


# ─────────────────────────────────────────────
# TASK 2 — MEDIUM: Standardize formats
# ─────────────────────────────────────────────
def medium_task():
    dirty_df = pd.DataFrame({
        "name":  ["alice", "BOB", "Carol", "DAVID", "eva"],
        "phone": ["9876543210", "98-765-43210", "(987)6543210", "9876543210", "987 654 3210"],
        "date":  ["2024-01-15", "15/01/2024", "Jan 15 2024", "2024-01-15", "01-15-2024"],
    })

    # Clean: name is Title Case, phone is XXXXXXXXXX (10 digits only), date is YYYY-MM-DD
    clean_df = pd.DataFrame({
        "name":  ["Alice", "Bob", "Carol", "David", "Eva"],
        "phone": ["9876543210", "9876543210", "9876543210", "9876543210", "9876543210"],
        "date":  ["2024-01-15", "2024-01-15", "2024-01-15", "2024-01-15", "2024-01-15"],
    })

    description = """
You are a data cleaning agent. You have a CSV dataset with INCONSISTENT FORMATS.

Problems in the data:
1. "name" column: mixed casing (all lowercase, ALL CAPS, etc.) → fix to Title Case
2. "phone" column: different formats (dashes, brackets, spaces) → fix to 10 digits only, no spaces or symbols
3. "date" column: different date formats → fix everything to YYYY-MM-DD format

To submit your answer, call the action: standardize
Your action string must look like this:
  standardize:name=title_case,phone=digits_only,date=iso

This tells the environment to apply those transformations to the dataset.
"""
    return {
        "dirty_df": dirty_df,
        "clean_df": clean_df,
        "description": description,
        "task_name": "medium",
    }


# ─────────────────────────────────────────────
# TASK 3 — HARD: Remove duplicates + outliers
# ─────────────────────────────────────────────
def hard_task():
    dirty_df = pd.DataFrame({
        "id":     [1, 2, 3, 2, 4, 5],
        "name":   ["Alice", "Bob", "Carol", "Bob", "David", "Eva"],
        "score":  [85, 90, 78, 90, 999, 88],   # 999 is an outlier
        "salary": [50000, 60000, 70000, 60000, 55000, 9999999],  # 9999999 is an outlier
    })

    # Clean: no duplicate ids, outliers replaced with column mean (of non-outlier values)
    clean_df = pd.DataFrame({
        "id":     [1, 2, 3, 4, 5],
        "name":   ["Alice", "Bob", "Carol", "David", "Eva"],
        "score":  [85, 90, 78, 85.25, 88],     # 999 replaced with mean of [85,90,78,88]=85.25
        "salary": [50000, 60000, 70000, 55000, 61000],  # 9999999 replaced with mean of others=59000... let's recalculate
    })

    # Recalculate clean values carefully
    # scores without outlier: [85, 90, 78, 88] → mean = 85.25
    # salaries without outlier: [50000, 60000, 70000, 55000] → mean = 58750
    clean_df = pd.DataFrame({
        "id":     [1, 2, 3, 4, 5],
        "name":   ["Alice", "Bob", "Carol", "David", "Eva"],
        "score":  [85.0, 90.0, 78.0, 85.25, 88.0],
        "salary": [50000.0, 60000.0, 70000.0, 55000.0, 58750.0],
    })

    description = """
You are a data cleaning agent. You have a CSV dataset with TWO problems:
1. DUPLICATE ROWS: Some rows share the same "id" — keep only the first occurrence.
2. OUTLIERS: Some numeric values are clearly wrong (e.g. score=999, salary=9999999).
   Replace outliers with the mean of the remaining valid values in that column.
   An outlier is any value more than 3 standard deviations from the mean.

To submit your answer, call the action: clean_hard
Your action string must look like this:
  clean_hard:remove_duplicates=true,replace_outliers=mean

This tells the environment to deduplicate by "id" and replace outliers with column mean.
"""
    return {
        "dirty_df": dirty_df,
        "clean_df": clean_df,
        "description": description,
        "task_name": "hard",
    }
