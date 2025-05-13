import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_dataframe(df):
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if "age" in df.columns:
        df["age"] = (df["age"] / 365).astype(int)

    if "gender" in df.columns:
        # Ensure gender is numeric before mapping
        if df["gender"].dtype == object:
            df["gender"] = df["gender"].map({"Female": 0, "Male": 1})
        else:
            df["gender"] = df["gender"].map({1: 0, 2: 1})

    if "height" in df.columns and "weight" in df.columns:
        df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

    # Preserve original values for visualization
    if "cholesterol" in df.columns:
        df["original_cholesterol"] = df["cholesterol"]
    if "gluc" in df.columns:
        df["original_gluc"] = df["gluc"]

    # Map text values back to numeric if needed
    value_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
    for col in ["cholesterol", "gluc"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map(value_map)

    df = df.dropna()

    cat_cols = []
    if "cholesterol" in df.columns:
        cat_cols.append("cholesterol")
    if "gluc" in df.columns:
        cat_cols.append("gluc")

    if cat_cols:
        for col in cat_cols:
            df[f"original_{col}"] = df[col]
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)

    scaler = StandardScaler()
    if "cardio" in df.columns:
        numeric_cols = [col for col in numeric_cols if col != "cardio"]
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)

    return df
