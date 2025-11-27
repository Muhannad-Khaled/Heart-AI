# ❤️ Heart AI – Advanced Healthcare Prediction Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Framework-Flask-black?style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

An advanced AI-powered healthcare web application designed to predict heart disease risks in real-time. This platform empowers healthcare professionals with fast, accurate, and insightful predictions for smarter clinical decisions and better patient outcomes.

> 🌐 **Access the Live Platform (Local):** The application will be running at [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

## 🎥 Project Demo

**[Demo: LinkedIn Post](https://www.linkedin.com/posts/muhannad-khaled_machinelearning-mlops-datascience-activity-7347882065586978819-fFCc?utm_source=share&utm_medium=member_desktop&rcm=ACoAADxRDisB2urrapLmHbxP9N_fisUH1-ANYzo)**

---

## 📚 Table of Contents
- [🚀 Features](#-features)
- [🧰 Tech Stack](#-tech-stack)
- [⚙️ Installation & Setup](#️-installation--setup)
- [📦 Project Structure](#-project-structure)
- [📸 Screenshots](#-screenshots)
- [👨‍💻 Developed By](#-developed-by)
- [📄 License](#-license)

---

## 🚀 Features

- 📈 **Predict:** Enter patient health data and receive AI-driven heart disease risk predictions.
- 📊 **Visualize:** Upload datasets and explore interactive visual insights and detailed statistics.
- 🎯 **Retrain:** Retrain the machine learning models with new datasets for improved accuracy and updated predictions.
- 🔐 **Authentication:** Secure login/logout functionality to protect sensitive user and prediction data.
- 📝 **Logging:** Comprehensive activity logging for monitoring prediction requests and system activity.

---

## 🧰 Tech Stack

- **Backend:** **Flask** (Python Web Framework)
- **Machine Learning:** **scikit-learn**, **pandas**, **numpy**, **joblib** (for model saving)
- **Frontend:** **HTML5**, **CSS3**, **Jinja2 Templates**
- **Visualization:** **Matplotlib**, **Seaborn**
- **Authentication:** Flask Session Management
- **Monitoring & Logging:** Python Logging Module

---

## ⚙️ Installation & Setup

Follow these steps to get a copy of the project running locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/Muhannad-Khaled/Heart-AI.git](https://github.com/Muhannad-Khaled/Heart-AI.git)
cd Heart-AI
```

### 2. Set up the Environment (Recommended)
```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment
source .venv/bin/activate       # macOS/Linux
# OR
.venv\Scripts\activate          # Windows
```

### 3. Install Dependencies
Install all required libraries using the provided `requirements.txt` file located in the `deployment` directory.

```bash
pip install -r deployment/requirements.txt
```

### 4. Run the Application
Execute the main application file.

```bash
python deployment/app.py
```

> **🎉 Success!** The application is now running. Open your web browser and navigate to:
> **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 📦 Project Structure

```text
healthcare-ai-app/
├── data/              # Storage for raw and processed datasets (e.g., CSVs)
├── deployment/        # Files necessary for running the application (app.py, requirements.txt, models)
├── modeling/          # Scripts for training, evaluating, and saving ML models
├── monitoring/        # Scripts for data quality and prediction logging/metrics
├── preprocessing/     # Scripts for cleaning and preparing data before modeling
├── static/            # Static files (CSS, JavaScript, Images) for the web interface
├── templates/         # Jinja2 HTML templates for the Flask application
├── .gitignore         # Files/directories ignored by Git
├── LICENSE            # Project license file
└── README.md          # Project documentation (this file)
```

---

## 📸 Screenshots

*(Add screenshots of the Prediction page, Visualization dashboard, and Authentication screen here.)*

---

## 👨‍💻 Developed By

| Role | Name | GitHub |
| :--- | :--- | :--- |
| **Lead Developer** | ✨ **Muhannad Khaled** | [Your GitHub Link](https://github.com/Muhannad-Khaled) |

---

## 📄 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for more details.

---

### ⭐️ **If you like this project, feel free to star it and share it with others!**
