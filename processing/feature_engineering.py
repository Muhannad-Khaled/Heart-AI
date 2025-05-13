def encode_features(df):
    """Apply one-hot encoding for categorical variables."""
    df = pd.get_dummies(df, columns=["cholesterol", "gluc"], drop_first=True)
    return df

