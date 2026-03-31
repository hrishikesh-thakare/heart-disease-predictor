# Heart Disease Risk Predictor (Bayesian Network Model)

A robust, interactive web application that predicts a patient's risk of heart disease using a **Causal Bayesian Network**. Unlike standard "black-box" machine learning models, this application leverages probabilistic modeling and **Explainable AI (XAI)** to not only predict the risk, but mathematically explain *why* the model made its prediction.

## Key Features

*   **Probabilistic Modeling:** Built using a Bayesian Network (`pgmpy`) that models the conditional dependencies between various patient vitals and health history factors.
*   **Explainable AI (XAI):** Automatically performs counterfactual reasoning (e.g., "What if this patient didn't smoke?") to quantify the exact percentage impact of individual risk factors on the total probability score.
*   **Interactive UI:** A highly responsive Streamlit dashboard featuring color-coded risk assessment headers (High, Medium, Low) and intuitive input forms.
*   **Transparent Output:** Exposes the raw probability breakdown $P(Disease=Yes)$ vs $P(Disease=No)$ for clinical transparency.

---

## Tech Stack

*   **Frontend:** [Streamlit](https://streamlit.io/) (for rapid, data-centric web UI generation)
*   **Modeling & Inference:** [pgmpy](https://pgmpy.org/) (for creating and querying the Discrete Bayesian Network)
*   **Data Processing:** `pandas`, `numpy`
*   **Model Persistence:** `pickle`

---

## How to Run the Project Locally

Follow these steps to set up the environment and run the application on your machine:

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Set up a Virtual Environment (Recommended)
Creating a virtual environment ensures that the project's dependencies are isolated from your system Python.

```bash
# Create the virtual environment
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Once the environment is active, install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
Start the Streamlit server to launch the frontend:

```bash
streamlit run app.py
```

### 5. Access the Dashboard
Once the server starts, it will automatically open the application in your default web browser. If it doesn't, navigate to `http://localhost:8501`.

---

## Project Structure

*   `app.py`: The main Streamlit web application script handling the UI, user inputs, XAI logic, and model inference.
*   `heart_disease_model.pkl`: The trained Causal Bayesian Network model, serialized for quick load times.
*   `heart-disease-model.ipynb`: The Jupyter Notebook used for the original exploratory data analysis (EDA), structure learning, and model training. 
*   `requirements.txt`: The list of python dependencies required to run the environment.
*   `heart_disease_model.xmlbif`: A legacy XMLBIF format of the model structure (kept for backward compatibility reference).

---

## How the Inference Works

1. **Evidence Collection:** The UI collects patient inputs (Age, Gender, Blood Pressure, Smoking History, etc.) and formats them into a binary evidence dictionary (e.g., `{"High_BP": "1.0", "Smoking": "0.0"}`).
2. **Variable Elimination:** The application uses exact inference via `VariableElimination` to calculate the marginal probability of the target node (`Disease`) given the provided evidence.
3. **Counterfactual Analysis:** To generate the "Risk Attribution" section, the application systematically flips active risk factors (e.g., changing "High BP" from "Yes" to "No") and re-queries the model to isolate and measure the absolute impact of each factor on the patient's resulting probability score.
