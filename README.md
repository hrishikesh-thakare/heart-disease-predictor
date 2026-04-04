# Heart Disease Risk Predictor (Dual-Model Comparative Dashboard)

A robust, interactive web application that evaluates a patient's risk of heart disease by comparing two distinct AI architectures: a **Causal Bayesian Network** and a **Random Forest**. This application bridges the gap between high-performance "black-box" prediction and clinical **Explainable AI (XAI)**.

## Key Features

*   **Model Comparison:** Side-by-side analysis of a **Random Forest** (optimized for predictive accuracy) vs. a **Bayesian Network** (optimized for causal reasoning).
*   **Explainable AI (XAI):** Uses the Bayesian Network to perform counterfactual reasoning (e.g., "What if this patient didn't smoke?") to quantify the exact percentage impact of individual risk factors.
*   **Dual-Inference Engine:** 
    *   **Random Forest:** Identifies complex, non-linear patterns across vitals.
    *   **Bayesian Network:** Maps the probabilistic dependencies between health factors.
*   **Interactive UI:** A professional Streamlit dashboard featuring color-coded risk assessment and intuitive input forms.
*   **Unified Input:** Both models accept the same 9 clinical features for a direct, fair comparison.

---

## Tech Stack

*   **Frontend:** [Streamlit](https://streamlit.io/)
*   **Harnessing PGMs:** [pgmpy](https://pgmpy.org/) (Bayesian Network inference)
*   **Machine Learning:** `scikit-learn` (Random Forest prediction)
*   **Data Handling:** `pandas`, `numpy`, `joblib`

---

## How to Run the Project Locally

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Set up a Virtual Environment
```bash
# Create the virtual environment
python -m venv venv

# Activate the environment (Windows)
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run heart_dashboard.py
```

---

## Project Structure

*   **`heart_dashboard.py`**: The main entry point for the Streamlit application.
*   **`bayesian_network.pkl`**: The trained Causal Bayesian Network model.
*   **`random_forest.pkl`**: The trained Random Forest model.
*   **`requirements.txt`**: Project dependencies.

---

## How the Inference Works

1.  **Unified Evidence Collection:** Both models utilize the same input features (Age, Gender, Blood Pressure, Smoking History, etc.).
2.  **Comparative Analysis:** 
    *   The **Random Forest** predicts the probability based on tree-based decision paths.
    *   The **Bayesian Network** uses **Variable Elimination** to calculate risk through probabilistic dependencies.
3.  **Risk Attribution (XAI):** The Bayesian Network isolates factors (e.g., changing "High BP" to "No") to measure the absolute impact of each risk factor on the patient's resulting probability score.
