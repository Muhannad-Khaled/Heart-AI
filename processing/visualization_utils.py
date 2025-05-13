import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid

def save_plot(fig, name=None):
    plot_dir = os.path.join("deployment", "static", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    if name is None:
        name = f"{uuid.uuid4().hex}.png"
    path = os.path.join(plot_dir, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return f"plots/{name}"

def analyze_target_balance(data: pd.DataFrame, target_column: str):
    if target_column not in data.columns:
        raise ValueError(f"Column '{target_column}' does not exist in the provided DataFrame.")
    
    target_counts = data[target_column].value_counts().dropna()
    if target_counts.empty:
        return None

    fig = plt.figure(figsize=(6, 4))
    target_counts.plot(kind='bar', edgecolor='black')
    plt.title(f"Target Distribution: {target_column}")
    plt.xlabel("Classes")
    plt.ylabel("Samples")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return save_plot(fig, f"{target_column}_balance.png")

def plot_numeric_distributions(data: pd.DataFrame, limit=3):
    plots = []
    numeric_cols = [col for col in data.select_dtypes(include="number").columns if col != "gender"]

    for col in numeric_cols[:limit]:
        col_data = data[col].dropna()
        if col_data.empty:
            continue
        fig = plt.figure()
        sns.histplot(col_data, kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plots.append(save_plot(fig, f"{col}_hist.png"))

    return plots

def plot_correlation_heatmap(data: pd.DataFrame):
    corr_data = data.copy()
    exclude_cols = ["smoke", "alco", "active", "original_gluc"]
    corr_data = corr_data.drop(columns=[col for col in exclude_cols if col in corr_data.columns], errors='ignore')
    corr_data = corr_data.select_dtypes(include=["number"]).fillna(0)

    if corr_data.shape[1] >= 2:
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(corr_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        return save_plot(fig, "corr_heatmap.png")
    return None

def plot_pie_chart(data: pd.DataFrame, column: str):
    if f"original_{column}" in data.columns:
        counts = data[f"original_{column}"].value_counts().dropna()
    elif column in data.columns:
        counts = data[column].value_counts().dropna()
    else:
        return None

    if column == "gluc":
        counts = counts.reindex([1, 2, 3], fill_value=0)

    if counts.sum() == 0:
        return None

    label_mapping = {
        "cholesterol": {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"},
        "gluc": {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"},
        "gender": {0: "Female", 1: "Male"}
    }

    # Ensure labels appear as readable text, not numbers
    labels = [label_mapping.get(column, {}).get(int(val), val) for val in counts.index]

    fig = plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
    plt.title(f"Pie Chart for {column.capitalize()}")
    plt.tight_layout()

    return save_plot(fig, f"{column}_pie.png")

def plot_boxplots(data: pd.DataFrame, limit=3):
    plots = []
    numeric_cols = [col for col in data.select_dtypes(include="number").columns if col != "gender"]

    for col in numeric_cols[:limit]:
        col_data = data[col].dropna()
        if col_data.empty:
            continue
        fig = plt.figure()
        sns.boxplot(y=col_data)
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plots.append(save_plot(fig, f"{col}_box.png"))

    return plots

def generate_dataset_info(data: pd.DataFrame):
    info = {
        "Number of Samples": [len(data)],
        "Number of Features": [len(data.columns)],
        "Missing Values": [data.isnull().sum().sum()],
        "Duplicate Rows": [data.duplicated().sum()],
        "Target Distribution (if exists)": [data["cardio"].value_counts().to_dict() if "cardio" in data.columns else "N/A"]
    }
    info_df = pd.DataFrame(info)
    return info_df.to_html(index=False, classes="table table-striped")
