import streamlit as st
import os
import sys
import pickle
import pgmpy.models
from pgmpy.inference import VariableElimination
import joblib
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Backwards compatibility for models saved with newer pgmpy versions
sys.modules['pgmpy.models.DiscreteBayesianNetwork'] = pgmpy.models
pgmpy.models.DiscreteBayesianNetwork = pgmpy.models.BayesianNetwork

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Load models (caching so it only loads once)
@st.cache_resource
def load_models():
    pgm_file = os.path.join(os.path.dirname(__file__), "bayesian_network.pkl")
    rf_file = os.path.join(os.path.dirname(__file__), "random_forest.pkl")
    
    infer = None
    rf_model = None
    rf_encoders = None
    
    try:
        if os.path.exists(pgm_file):
            with open(pgm_file, 'rb') as f:
                model = pickle.load(f)
            infer = VariableElimination(model)
    except Exception as e:
        st.error(f"Error loading PGM model: {e}")
        
    try:
        if os.path.exists(rf_file):
            rf_dict = joblib.load(rf_file)
            rf_model = rf_dict['model']
            rf_encoders = rf_dict['encoders']
    except Exception as e:
        st.error(f"Error loading Random Forest model: {e}")
        
    return infer, rf_model, rf_encoders

infer, rf_model, rf_encoders = load_models()

# User-friendly feature labels for explanations
FEATURE_LABELS = {
    "High_BP": "High Blood Pressure",
    "High_Cholesterol": "High Cholesterol",
    "Diabetes": "Diabetes",
    "Smoking": "Smoking History",
    "Obesity": "Obesity",
    "Family_History": "Family History",
    "Chest_Pain": "Chest Pain"
}

def calculate_contributions(infer, evidence, current_risk, target="Disease"):
    """Calculate how much each 'Yes' factor contributed to the risk probability."""
    contributions = []
    
    # We only analyze 'Yes' (1.0) factors to see their impact
    for feature, value in evidence.items():
        if value == "1.0" and feature in FEATURE_LABELS:
            # Create a 'What-If' evidence where this factor is 'No'
            counterfactual_evidence = evidence.copy()
            counterfactual_evidence[feature] = "0.0"
            
            try:
                # Query the model for the new risk
                cf_result = infer.query(variables=[target], evidence=counterfactual_evidence)
                state_names = cf_result.state_names[target]
                idx_1_0 = state_names.index('1.0')
                new_risk = cf_result.values[idx_1_0] * 100
                
                # Contribution is the difference
                impact = current_risk - new_risk
                if impact > 0.1: # Significant impact threshold
                    contributions.append((FEATURE_LABELS[feature], impact))
            except:
                continue
                
    # Sort by impact
    return sorted(contributions, key=lambda x: x[1], reverse=True)

# Custom CSS for premium aesthetic
st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(135deg, #ef4444, #b91c1c);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
    }
    .css-1d391kg {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    h1, h2, h3 {
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    .stSelectbox label, .stRadio label {
        color: #cbd5e1;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Heart Disease Risk Predictor")
st.markdown("Use this interactive tool to assess your risk based on a Bayesian Network model.")

if not infer:
    st.stop()

# Helper dictionaries to map UI choices back to BIF node string formats
yes_no_mapping = {"No": "0.0", "Yes": "1.0"}
age_mapping = {"Young": "Young", "Middle Aged": "Middle-Aged", "Senior": "Senior"}
gender_mapping = {"Female": "0.0", "Male": "1.0"}

with st.form("prediction_form"):
    st.subheader("Patient Vitals & History")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age_input = st.selectbox("Age Group", ["Young", "Middle Aged", "Senior"])
        gender_input = st.selectbox("Gender", ["Male", "Female"])
        chest_pain = st.radio("Experiencing Chest Pain?", ["No", "Yes"], horizontal=True)
        high_bp = st.radio("High Blood Pressure?", ["No", "Yes"], horizontal=True)
        high_chol = st.radio("High Cholesterol?", ["No", "Yes"], horizontal=True)
        
    with col2:
        diabetes = st.radio("Diabetes?", ["No", "Yes"], horizontal=True)
        obesity = st.radio("Obesity?", ["No", "Yes"], horizontal=True)
        smoking = st.radio("Smoking?", ["No", "Yes"], horizontal=True)
        family_hist = st.radio("Family History of Heart Disease?", ["No", "Yes"], horizontal=True)

    submit = st.form_submit_button("Predict Risk")

if submit:
    # Prepare evidence dictionary
    evidence = {
        "Age": age_mapping[age_input],
        "Gender": gender_mapping[gender_input],
        "Chest_Pain": yes_no_mapping[chest_pain],
        "High_BP": yes_no_mapping[high_bp],
        "High_Cholesterol": yes_no_mapping[high_chol],
        "Diabetes": yes_no_mapping[diabetes],
        "Obesity": yes_no_mapping[obesity],
        "Smoking": yes_no_mapping[smoking],
        "Family_History": yes_no_mapping[family_hist]
    }
    
    with st.spinner("Analyzing Models..."):
        try:
            # 1. PGM Query
            result = infer.query(variables=["Disease"], evidence=evidence)
            state_names = result.state_names['Disease']
            idx_1_0 = state_names.index('1.0')
            idx_0_0 = state_names.index('0.0')
            
            pgm_risk_prob_val = result.values[idx_1_0]
            pgm_no_risk_prob_val = result.values[idx_0_0]
            pgm_risk_prob = pgm_risk_prob_val * 100
            
            # 2. Random Forest Query
            rf_risk_prob = None
            if rf_model and rf_encoders:
                # Dynamically fetch expected features to stay synced with model retrains
                rf_features = list(rf_model.feature_names_in_) if hasattr(rf_model, 'feature_names_in_') else list(evidence.keys())
                
                rf_input = {}
                for col in rf_features:
                    if col in evidence:
                        rf_input[col] = evidence[col]
                    else:
                        rf_input[col] = "0.0" # Safe default if any feature ever goes missing
                        
                # Encode inputs
                encoded_input = {}
                for col in rf_features:
                    try:
                        # Try exact match first
                        encoded_input[col] = rf_encoders[col].transform([rf_input[col]])[0]
                    except ValueError:
                        # If the user's encoder expects a float or int instead of a string
                        try:
                            encoded_input[col] = rf_encoders[col].transform([float(rf_input[col])])[0]
                        except:
                            encoded_input[col] = 0
                    except Exception:
                        encoded_input[col] = 0
                        
                df_input = pd.DataFrame([encoded_input], columns=rf_features)
                try:
                    rf_probs = rf_model.predict_proba(df_input)[0]
                    rf_risk_prob = rf_probs[-1] * 100 # Assuming last class is positive
                except Exception as e:
                    # In case of prediction issues (e.g., shape mismatch)
                    rf_risk_prob = None
            
            st.markdown("---")
            st.subheader("Model Comparison: Bayesian Network vs Random Forest")
            
            # Display Side-by-Side Model Predictions
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.markdown("### Random Forest")
                st.markdown("**Focus:** Predictive Accuracy")
                if rf_risk_prob is not None:
                    if rf_risk_prob > 50:
                        st.error(f"**High Risk:** {rf_risk_prob:.2f}%")
                        st.progress(int(rf_risk_prob))
                    elif rf_risk_prob > 20:
                        st.warning(f"**Medium Risk:** {rf_risk_prob:.2f}%")
                        st.progress(int(rf_risk_prob))
                    else:
                        st.success(f"**Low Risk:** {rf_risk_prob:.2f}%")
                        st.progress(int(rf_risk_prob))
                else:
                    st.write("Model unavailable.")

            with comp_col2:
                st.markdown("### Bayesian Network")
                st.markdown("**Focus:** Causal Inference & Explainability")
                if pgm_risk_prob > 50:
                    st.error(f"**High Risk:** {pgm_risk_prob:.2f}%")
                    st.progress(int(pgm_risk_prob))
                elif pgm_risk_prob > 20:
                    st.warning(f"**Medium Risk:** {pgm_risk_prob:.2f}%")
                    st.progress(int(pgm_risk_prob))
                else:
                    st.success(f"**Low Risk:** {pgm_risk_prob:.2f}%")
                    st.progress(int(pgm_risk_prob))
            
            st.markdown("---")
            st.markdown("### Why Compare Them?")
            st.info('''
            **The Random Forest (Black Box)** may offer relatively higher predictive accuracy by identifying complex, non-linear patterns across many variables. However, it cannot easily explain *why* it made a prediction. 
            
            **The Bayesian Network (White Box/PGM)**, even if it has lower raw accuracy, mathematically maps the relationships between variables. **It excels in Explainability (XAI)**—allowing us to see exactly which factor is driving your risk up or down by performing counterfactual "What-If" analysis. 
            ''')


            # Explainable AI: Key Contributors from BN
            contributions = calculate_contributions(infer, evidence, pgm_risk_prob)
            
            if contributions:
                st.markdown("---")
                st.subheader("Analysis: Risk Level Attribution")
                st.write("The following factors were identified as the strongest contributors to your risk profile:")
                
                for label, impact in contributions:
                    # Determine severity prefix based on impact
                    if impact > 10:
                        prefix = "Critical Factor:"
                    elif impact > 5:
                        prefix = "Significant Factor:"
                    else:
                        prefix = "Contributing Factor:"
                        
                    st.markdown(f"**{prefix} {label}:** Added approximately **{impact:.1f}%** to the overall risk score.")

            with st.expander("View Case Details"):
                # Define readable values mapping
                val_map = {"1.0": "Yes", "0.0": "No"}
                
                # We specifically process Gender since its labels are Male/Female
                gender_val = "Male" if evidence["Gender"] == "1.0" else "Female"
                
                st.write(f"**Age Group:** {evidence['Age']}")
                st.write(f"**Gender:** {gender_val}")
                
                # Loop through all other binary factors
                for feature, model_val in evidence.items():
                    if feature not in ["Age", "Gender"] and feature in FEATURE_LABELS:
                        readable_val = val_map.get(model_val, model_val)
                        st.write(f"**{FEATURE_LABELS[feature]}:** {readable_val}")
                
        except Exception as e:
            st.error(f"An error occurred during inference: {e}")
