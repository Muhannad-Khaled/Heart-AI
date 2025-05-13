import pandas as pd
import os

def preprocess_data(input_path="data/raw/cardio_data.csv", output_path="data/processed/processed_data.csv"):
    # 1️⃣ Load Data with Correct Delimiter
    df = pd.read_csv(input_path, delimiter=",")
    print(f"📌 Initial Columns: {df.columns.tolist()}")

    # 2️⃣ Drop 'id' column if exists
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({1: 0, 2: 1})
    
    df = pd.get_dummies(df, columns=["cholesterol", "gluc"], drop_first=True)

    # 3️⃣ Remove Missing Values
    df = df.dropna()

    # 4️⃣ Remove Outliers using IQR
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # 5️⃣ Clean String Columns (remove spaces and commas)
    str_cols = df.select_dtypes(include="object").columns.tolist()
    for col in str_cols:
        df[col] = df[col].str.strip().str.replace(",", "", regex=False)

    # 6️⃣ Reset index and add 'id' column explicitly
    df = df.reset_index(drop=True)
    df.insert(0, "id", df.index)

    # 7️⃣ Save Processed Data Correctly (as clean CSV)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Processed data saved to: {output_path}")
    return df

if __name__ == "__main__":
    preprocess_data()


import pandas as pd
import os
from processing.preprocessing_utils import preprocess_dataframe

def run_preprocessing(input_path="data/raw/cardio_data.csv", output_path="data/processed/processed_data.csv"):
    try:
        # ✅ Load raw data
        print("📥 Loading raw data...")
        df = pd.read_csv(input_path, delimiter=",")

        # ✅ Preprocess data using the existing function
        print("⚙️  Applying preprocessing steps...")
        processed_df = preprocess_dataframe(df)

        # ✅ Save processed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processed_df.to_csv(output_path, index=False)
        print(f"✅ Processed data saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")

if __name__ == "__main__":
    run_preprocessing()
