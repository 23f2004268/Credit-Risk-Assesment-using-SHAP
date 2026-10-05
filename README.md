# 💳 Credit Ledger — Loan Default Risk Assessment

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11.9-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-orange?style=for-the-badge)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)

**An underwriting desk for consumer loan applications — powered by XGBoost and SHAP.**

[Live Demo](#) · [API Docs](#api-reference) · [Report a Bug](../../issues)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [API Reference](#-api-reference)
- [Getting Started](#-getting-started)
- [Deployment](#-deployment)
- [Dataset](#-dataset)
- [Results](#-results)
- [License](#-license)

---

## 🔍 Overview

**Credit Ledger** is a full-stack machine learning web application that predicts the probability of a loan applicant defaulting on their loan. It combines a trained **XGBoost classifier** with a clean, dark-themed UI to simulate an underwriting desk experience.

The model was trained on **32,581 real-world loan records** and uses a **custom-optimized classification threshold** (instead of the default 0.5) to better handle the class imbalance inherent in credit risk data.

> ⚠️ **Disclaimer:** Figures are model estimates, not actual lending decisions. This project is for educational and demonstration purposes only.

---

## 🎬 Demo

> *(Add a screenshot or GIF of your app here)*

```
┌─────────────────────────────────────────────┐
│  Credit Ledger  ●  service ready            │
├─────────────────────────────────────────────┤
│  01 — Applicant                             │
│  02 — Loan Request                          │
│  03 — Credit Bureau File                    │
│                                             │
│         [ Assess Risk ]                     │
├─────────────────────────────────────────────┤
│   ◉ 34.2%          ┌──────────┐             │
│   default prob     │ LOW RISK │             │
│                    └──────────┘             │
└─────────────────────────────────────────────┘
```

---

## ✨ Features

- 🤖 **XGBoost ML Model** — trained on 32K+ real loan records
- 📊 **Custom Threshold** — optimized beyond the default 0.5 to handle class imbalance
- 🎨 **Animated UI** — circular gauge, stamp animation, and live number counter
- ⚡ **FastAPI Backend** — fast, async REST API with Pydantic validation
- 📱 **Responsive Design** — works on desktop and mobile
- 🔄 **Auto loan-to-income ratio** — calculated in real-time from income and loan amount
- 🟢 **Service health indicator** — live API status shown in the header

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **ML Model** | XGBoost 3.4.1, scikit-learn 1.6.1 |
| **Explainability** | SHAP (notebook analysis) |
| **Backend** | FastAPI 0.115.0, Uvicorn, Pydantic v2 |
| **Data** | Pandas 2.2.2 |
| **Serialization** | Joblib 1.4.2 |
| **Frontend** | Vanilla JS, HTML5, CSS3 |
| **Fonts** | Source Serif 4, IBM Plex Sans, IBM Plex Mono |
| **Deployment** | Render.com |
| **Language** | Python 3.11.9 |

---

## 📁 Project Structure

```
Credit-Risk-Assesment-using-SHAP/
│
├── 📓 Credit_Risk.ipynb          # ML pipeline: EDA, preprocessing, training, SHAP
├── 📊 credit_risk_dataset.csv    # Raw dataset (32,581 loan records)
│
├── 🤖 credit_risk_model.pkl      # Trained XGBoost pipeline (serialized)
├── 🎯 best_threshold.pkl         # Optimized classification threshold
│
├── 🚀 main.py                    # FastAPI application
│
├── static/
│   ├── 🌐 index.html             # Frontend UI
│   ├── ⚙️  script.js             # Frontend logic & API calls
│   └── 🎨 style.css              # Dark theme styling
│
├── 📋 requirements.txt           # Python dependencies
├── ☁️  render.yaml               # Render.com deployment config
└── 🐍 runtime.txt                # Python version (3.11.9)
```

---

## 🧠 ML Pipeline

### Dataset

The model is trained on the [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) with **32,581 records** and **12 features**.

| Feature | Type | Description |
|---|---|---|
| `person_age` | int | Applicant age |
| `person_income` | float | Annual income |
| `person_home_ownership` | categorical | RENT / MORTGAGE / OWN / OTHER |
| `person_emp_length` | float | Employment length (years) |
| `loan_intent` | categorical | PERSONAL / EDUCATION / MEDICAL / VENTURE / HOMEIMPROVEMENT / DEBTCONSOLIDATION |
| `loan_grade` | categorical | A / B / C / D / E / F / G |
| `loan_amnt` | float | Loan amount |
| `loan_int_rate` | float | Interest rate (%) |
| `loan_percent_income` | float | Loan amount as a ratio of income |
| `cb_person_default_on_file` | categorical | Prior default on record (Y/N) |
| `cb_person_cred_hist_length` | int | Credit history length (years) |
| `loan_status` | int | **Target** — 0 = repaid, 1 = defaulted |

### Class Distribution

```
Non-default (0):  25,473  (~78%)
Default     (1):   7,108  (~22%)
```

> The dataset is **imbalanced**, which is why a custom threshold is used instead of the default 0.5.

### Pipeline Steps

```
Raw Data
   │
   ▼
EDA & Outlier Detection
   │  (person_age > 100, person_emp_length > 60)
   ▼
Missing Value Imputation
   │  (loan_int_rate: 3,116 missing, person_emp_length: 895 missing)
   ▼
Categorical Encoding
   │
   ▼
XGBoost Classifier
   │
   ▼
Threshold Optimization
   │  (best_threshold.pkl — optimized for imbalanced classes)
   ▼
SHAP Explainability Analysis
```

### Model Artifacts

| File | Description |
|---|---|
| `credit_risk_model.pkl` | Full sklearn Pipeline (preprocessing + XGBoost) |
| `best_threshold.pkl` | Optimal decision threshold (float) |

---

## 📡 API Reference

### `POST /predict`

Predicts the default probability for a loan application.

**Request Body**

```json
{
  "person_age": 30,
  "person_income": 600000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 100000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 6
}
```

**Response**

```json
{
  "default_probability": 0.342,
  "default_prediction": 0,
  "threshold": 0.45,
  "Result": "Low Risk"
}
```

| Field | Type | Description |
|---|---|---|
| `default_probability` | float | Probability of default (0–1) |
| `default_prediction` | int | 0 = Low Risk, 1 = High Risk |
| `threshold` | float | Decision threshold used |
| `Result` | string | `"High Risk"` or `"Low Risk"` |

**Interactive API Docs** (auto-generated by FastAPI):
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/credit-risk-assessment.git
cd credit-risk-assessment

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
uvicorn main:app --reload
```

Then open your browser at:
- **App:** [http://localhost:8000](http://localhost:8000)
- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Re-training the Model

Open and run `Credit_Risk.ipynb` in Jupyter:

```bash
pip install jupyter shap matplotlib seaborn
jupyter notebook Credit_Risk.ipynb
```

This will regenerate `credit_risk_model.pkl` and `best_threshold.pkl`.

---

## ☁️ Deployment

This project is deployed on **[Render.com](https://credit-risk-assesment-using-shap-1-ok7w.onrender.com)** using the configuration in `render.yaml`.

```yaml
services:
  - type: web
    name: credit-ledger
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    autoDeploy: true
```

### Deploy Your Own

1. Fork this repository
2. Create a new **Web Service** on Render
3. Connect your GitHub repo
4. Render will auto-detect `render.yaml` and deploy

> **Note:** The free tier spins down after inactivity. The first request after a period of sleep may take 30–60 seconds.

---

## 📊 Dataset

- **Source:** [Credit Risk Dataset — Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Records:** 32,581
- **Features:** 11 input features + 1 target
- **Missing values:** `loan_int_rate` (9.6%), `person_emp_length` (2.7%)

---

## 📈 Results

| Metric | Value |
|---|---|
| Model | XGBoost Classifier |
| Training set size | ~26,000 records |
| Test set size | ~6,500 records |
| Custom threshold | Optimized (see `best_threshold.pkl`) |

> Full evaluation metrics (AUC-ROC, F1, Precision, Recall) are available in `Credit_Risk.ipynb`.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) by Laotse on Kaggle
- [FastAPI](https://fastapi.tiangolo.com/) — modern, fast web framework for Python
- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosting library
- [SHAP](https://shap.readthedocs.io/) — explainable AI library
- [IBM Plex](https://www.ibm.com/plex/) — open-source font family

---

<div align="center">
  Made with ❤️ | <a href="https://github.com/23f2004268">@23f2004268</a>
</div>
