def clean_data(df):
    """Clean missing values and remove outliers using IQR."""
    # 1️⃣ Fill missing values
    df = df.fillna(df.median(numeric_only=True))

    # 2️⃣ Remove Outliers for numerical features using IQR
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df
