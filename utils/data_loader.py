import pandas as pd

def load_data(path="data/raw/cardio_data.csv"):
    """Load the new cardio dataset and handle preprocessing basics."""
    df = df.drop(columns=["id"])
    df["age"] = (df["age"] / 365).astype(int)  # تحويل العمر إلى سنوات
    df["gender"] = df["gender"].map({1: 0, 2: 1})  # 0: Female, 1: Male
    return df

