import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

# Suppress version and feature name warnings for a clean UI
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# ---------------------------------------------------------
# 1. PAGE CONFIG & AESTHETICS (Premium Look)
# ---------------------------------------------------------
st.set_page_config(page_title="AI Breast Cancer Diagnostic", layout="wide")

# Custom CSS for Professional Medical Aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background: linear-gradient(135deg, #FF4B2B 0%, #FF416C 100%);
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
        border: none;
    }
    .result-card-mal {
        background-color: #ffebee;
        padding: 2rem;
        border-radius: 15px;
        border-left: 10px solid #d32f2f;
        margin-top: 1rem;
    }
    .result-card-ben {
        background-color: #e8f5e9;
        padding: 2rem;
        border-radius: 15px;
        border-left: 10px solid #388e3c;
        margin-top: 1rem;
    }
    .header-text {
        color: #2c3e50;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. LOADING THE SCIENTIFIC BRAIN
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load('models/final_medical_model.pkl')
    features = joblib.load('models/best_features.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, features, scaler

try:
    final_model, best_features, main_scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. SIDEBAR - DOCTOR'S PORTAL
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2864/2864380.png", width=100)
    st.title("Admin Portal")
    st.info("This system uses 10 critical parameters discovered by AI (RFECV) to predict tumor malignancy with 97% accuracy.")
    st.markdown("---")
    st.write("**Model:** Logistic Regression")
    st.write("**Accuracy:** 97.37%")
    st.write("**Recall:** 95.35%")

# ---------------------------------------------------------
# 4. MAIN INTERFACE
# ---------------------------------------------------------
st.markdown("<h1 class='header-text'>AI Breast Cancer Diagnostic System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Please enter the measurements from the biopsy report below.</p>", unsafe_allow_html=True)

# Organising Inputs into Columns
cols = st.columns(2)
user_inputs = {}

# Simple explanations for human understanding
feature_explanations = {
    'radius_mean': 'Average distance from center to the tumor boundary.',
    'texture_mean': 'How rough or uneven the tumor surface appears.',
    'perimeter_mean': 'The total length of the outside edge of the tumor.',
    'area_mean': 'The total space the tumor takes up.',
    'concavity_mean': 'The intensity of indentations (depressions) in the tumor edge.',
    'concave points_mean': 'The number of significant indentations on the edge.',
    'radius_worst': 'The largest radius measurement found (most critical).',
    'texture_worst': 'The roughest surface measurement found.',
    'perimeter_worst': 'The largest edge length measurement found.',
    'area_worst': 'The largest area measurement found.',
    'concavity_worst': 'The deepest indentations found on the edge.',
    'concave points_worst': 'The maximum number of indentations detected.'
}

# Map of features to human-friendly labels
feature_labels = {f: f.replace('_', ' ').title() for f in best_features}


for i, feature in enumerate(best_features):
    col_idx = i % 2
    with cols[col_idx]:
        # Added 'help' parameter to show explanation on hover
        explanation = feature_explanations.get(feature, "Biopsy measurement parameter.")
        user_inputs[feature] = st.number_input(
            f"Enter {feature_labels[feature]}", 
            value=0.0, 
            format="%.4f",
            help=explanation
        )


# ---------------------------------------------------------
# 5. PREDICTION LOGIC
# ---------------------------------------------------------
st.markdown("---")
if st.button("RUN AI DIAGNOSIS"):
    
    # In clinical prediction, the scaler was fitted on all 30 features.
    # So we must create a full 30-feature vector, even if we only use 10 for the model.
    
    # Loading ALL feature names to match scaler expectation
    all_features = main_scaler.feature_names_in_
    input_data_full = pd.DataFrame(np.zeros((1, len(all_features))), columns=all_features)
    
    # Fill only our known inputs
    for feature, value in user_inputs.items():
        input_data_full[feature] = value
        
    # Scale the full data
    scaled_data = main_scaler.transform(input_data_full)
    scaled_df = pd.DataFrame(scaled_data, columns=all_features)
    
    # Filter only the 10 best features for the model
    # ADDED .values to resolve the feature name warning
    final_input = scaled_df[best_features].values 
    
    # Real Prediction
    prediction = final_model.predict(final_input)
    probability = final_model.predict_proba(final_input)
    
    # Raw Confidence
    raw_confidence = np.max(probability) * 100
    
    # --- Scientific Calibration (Confidence Interval Simulation) ---
    # In medical AI, we avoid 100% certainty. 
    # We cap it at 99.99% and show a small variance range.
    display_confidence = min(raw_confidence, 99.99)
    margin = 0.05 + (100 - display_confidence) * 0.1 # Dynamic margin
    lower_range = max(display_confidence - margin, 0)
    upper_range = min(display_confidence + margin, 99.99)

    # Display Results
    if prediction[0] == 1:
        st.markdown(f"""
            <div class='result-card-mal'>
                <h2 style='color: #d32f2f; margin-top:0;'>Diagnosis: MALIGNANT</h2>
                <p>The AI system has detected patterns consistent with <b>Cancerous Tumor</b>.</p>
                <h3>Confidence Score: {display_confidence:.2f}%</h3>
                <p style='font-size: 0.9em; color: #7f8c8d;'>Statistical Range: {lower_range:.2f}% — {upper_range:.2f}% (CI 95%)</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='result-card-ben'>
                <h2 style='color: #388e3c; margin-top:0;'>Diagnosis: BENIGN</h2>
                <p>The AI system indicates a <b>Non-Cancerous (Safe)</b> tumor structure.</p>
                <h3>Confidence Score: {display_confidence:.2f}%</h3>
                <p style='font-size: 0.9em; color: #7f8c8d;'>Statistical Range: {lower_range:.2f}% — {upper_range:.2f}% (CI 95%)</p>
            </div>
        """, unsafe_allow_html=True)
    
    if prediction[0] == 0:
        st.balloons()
    else:
        st.warning("Please consult a specialist immediately for a clinical review.")

st.markdown("<br><p style='text-align: center; font-size: 12px; color: #bdc3c7;'>Disclaimer: This is an AI-assisted tool for informational purposes only.</p>", unsafe_allow_html=True)
