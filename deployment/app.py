import os
import pandas as pd
import joblib
from flask import session
from monitoring.logging import log_prediction
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from modeling.train_model import train_model
from config.config import MODEL_PATH
from processing.preprocessing_utils import preprocess_dataframe
from processing.visualization_utils import (
    analyze_target_balance,
    plot_numeric_distributions,
    plot_correlation_heatmap,
    plot_pie_chart,
    plot_boxplots,
    generate_dataset_info
)

app = Flask(__name__)
app.secret_key = "secret"

# ✅ Load model and expected features
model = joblib.load(MODEL_PATH)
FEATURE_PATH = os.path.join("modeling", "features.pkl")
expected_features = joblib.load(FEATURE_PATH) if os.path.exists(FEATURE_PATH) else []

# ✅ Human-readable labels & dropdown options
field_labels = {
    "age": "Age (Years)",
    "gender": "Gender",
    "height": "Height (cm)",
    "weight": "Weight (kg)",
    "ap_hi": "Systolic BP",
    "ap_lo": "Diastolic BP",
    "cholesterol": "Cholesterol Level",
    "gluc": "Glucose Level",
    "smoke": "Smoking",
    "alco": "Alcohol Consumption",
    "active": "Physical Activity"
}

field_options = {
    "gender": {0: "Female", 1: "Male"},
    "cholesterol": {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"},
    "gluc": {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"},
    "smoke": {0: "No", 1: "Yes"},
    "alco": {0: "No", 1: "Yes"},
    "active": {0: "No", 1: "Yes"}
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    user_input = {}
    if request.method == "POST":
        try:
            raw_input = {}
            for col in field_labels.keys():
                value = request.form.get(col)
                if value is None or value == "":
                    raise ValueError(f"Missing value for {col}")

                # Handle categorical mappings
                if col in field_options:
                    reverse_mapping = {v: k for k, v in field_options[col].items()}
                    mapped_value = reverse_mapping.get(value)
                    if mapped_value is None:
                        raise ValueError(f"Invalid value for {col}: {value}")
                    raw_input[col] = float(mapped_value)
                else:
                    raw_input[col] = float(value)

            user_input = {
                field_labels.get(col, col): field_options.get(col, {}).get(int(raw_input[col]), raw_input[col])
                if col in field_options else raw_input[col]
                for col in field_labels.keys()
            }

            df = pd.DataFrame([raw_input])
            # df = preprocess_dataframe(df)

            for col in expected_features:
                if col not in df.columns:
                    df[col] = 0
            df = df[expected_features]

            result = int(model.predict(df)[0])
            prediction = "The patient is at risk!" if result == 1 else "The patient is NOT at risk."
            log_prediction(raw_input, result)
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template(
        "index.html",
        columns=field_labels.keys(),
        prediction=prediction,
        user_input=user_input,
        field_labels=field_labels,
        field_options=field_options
    )

@app.route("/visualize", methods=["GET", "POST"])
def visualize():
    summary = None
    plot_paths = []
    dataset_info = None

    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded.")
            return redirect(request.url)

        file = request.files["file"]
        try:
            df = pd.read_csv(file, delimiter=",")
            df = preprocess_dataframe(df)

            summary = {
                "head": df.head().to_html(classes="table table-striped"),
                "missing": df.isnull().sum().to_frame("Missing").to_html(classes="table table-striped"),
                "describe": df.describe().to_html(classes="table table-striped")
            }

            dataset_info = generate_dataset_info(df)

            if "target" in df.columns or "cardio" in df.columns:
                target_col = "target" if "target" in df.columns else "cardio"
                plot_paths.append(analyze_target_balance(df, target_col))

            plot_paths += plot_numeric_distributions(df, limit=3)

            heatmap_plot = plot_correlation_heatmap(df)
            if heatmap_plot:
                plot_paths.append(heatmap_plot)

            for col in ["gender", "original_cholesterol", "original_gluc"]:
                if col in df.columns:
                    display_col = col.replace("original_", "")
                    pie_plot = plot_pie_chart(df, display_col)
                    if pie_plot:
                        plot_paths.append(pie_plot)

            plot_paths += plot_boxplots(df, limit=3)

        except Exception as e:
            flash(f"Error processing file: {e}")

    return render_template(
        "visualize.html",
        summary=summary,
        plots=plot_paths,
        dataset_info=dataset_info
    )

@app.route("/train", methods=["GET", "POST"])
def train():
    message = None
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded.")
            return redirect(request.url)

        file = request.files["file"]
        try:
            df = pd.read_csv(file, delimiter=",")
            df = preprocess_dataframe(df)

            train_model(df)

            global expected_features
            if os.path.exists(FEATURE_PATH):
                expected_features = joblib.load(FEATURE_PATH)

            message = "✅ Model retrained and saved successfully."
        except Exception as e:
            message = f"❌ Error: {e}"

    return render_template("training.html", message=message)

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        df = preprocess_dataframe(df)

        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_features]

        prediction = int(model.predict(df)[0])
        log_prediction(data, prediction)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


app.secret_key = "super_secret_key"  # Already present, just ensure it's there.


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # For demo, accept any email/password (you can implement real auth later)
        if email and password:
            session['logged_in'] = True
            flash("✅ Login successful!", "success")
            return redirect(url_for("index"))
        else:
            flash("❌ Invalid credentials!", "error")
    return render_template("login.html", title="Login")

@app.before_request
def require_login():
    if request.endpoint not in ('login', 'static') and not session.get('logged_in'):
        return redirect(url_for('login'))

@app.route("/logout")
def logout():
    session.clear()
    flash("👋 Logged out successfully!")
    return redirect(url_for('login'))



@app.route("/about")
def about():
    return render_template("about.html", title="About")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")
        # 📨 هنا تقدر تخزن الرسالة في قاعدة بيانات أو ترسلها على إيميل
        print(f"📧 Message Received: Name={name}, Email={email}, Message={message}")
        flash("✅ Message sent successfully!", "success")
        return redirect(url_for("contact"))

    return render_template("contact.html", title="Contact Us")


if __name__ == "__main__":
    app.run(debug=True)
