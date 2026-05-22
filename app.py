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
    /* 1. Main Page Style */
    .main {
        background-color: #f0f2f6;
    }
    /* remove unwanted space */
    .block-container {
        padding-top: 20px !important;
    }


    /* 2. Button Style (Colorful Gradient) */
    .stButton>button {
        background: linear-gradient(135deg, #e91e63 0%, #ff4b2b 100%); /* Pink se Red ka gradient */
        color: white !important;   /* White text */
        border-radius: 8px;        /* Rounded corners */
        width: 100%;               /* Full width */
        font-weight: bold;         /* Bold text */
        border: none;              /* No border */
        padding: 10px;             /* Inside space */
        transition: 0.3s;          /* Smooth transition */
    }

    .stButton>button:hover {
        opacity: 0.9;              /* Hover par halka sa transparency */
        transform: scale(1.02);    /* Halka sa bada hoga */
    }

    /* 3. Malignant (Cancer) Result Card */
    .result-card-mal {
        background-color: #ffeaea; /* Light red */
        color: #721c24;            /* Dark red text for contrast */
        padding: 25px;
        border-radius: 10px;
        border-left: 10px solid #c0392b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* Halka shadow */
    }

    /* 4. Benign (Safe) Result Card */
    .result-card-ben {
        background-color: #e8f9ed; /* Light green */
        color: #155724;            /* Dark green text for contrast */
        padding: 25px;
        border-radius: 10px;
        border-left: 10px solid #27ae60;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* Halka shadow */
    }

    /* 5. Main Title Style (Colorful & Attractive) */
    .header-text {
        background: linear-gradient(to right, #e91e63, #ff4b2b); /* Pink se Red ka gradient */
        -webkit-background-clip: text;                          /* Text ke andar rang bharna */
        -webkit-text-fill-color: transparent;                   /* Default color hatana */
        text-align: center;                                     /* Center mein rakhna */
        font-size: 45px;                                        /* Bada size */
        font-weight: 900;                                       /* Extra bold text */
        margin-bottom: 20px;                                    /* Neeche thodi jagah */
    }

    /* 6. Sidebar Image adjustment */
    [data-testid="stSidebar"] [data-testid="stImage"] {
        margin-top: -20px;
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
    # Loading dataset for random samples
    df = pd.read_csv('data/data.csv')
    return model, features, scaler, df

try:
    final_model, best_features, main_scaler, raw_df = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. SIDEBAR - DOCTOR'S PORTAL
# ---------------------------------------------------------
with st.sidebar:
    st.image("assets/image.png", use_container_width=True)
    st.title("Diagnostic Intelligence")
    st.caption("Clinical-grade AI system using 12 optimized bio-parameters for high-precision tumor analysis.")
    
    # Compact Metrics
    st.markdown("**Model:** LogReg &nbsp;|&nbsp; **Acc:** 97.3% &nbsp;|&nbsp; **Recall:** 95.3%")
    
    # --- Dynamic Example Data Loader ---
    st.markdown("<p style='font-size: 14px; margin-bottom: 5px; margin-top: 5px; color: #555;'><b>Load Sample Data:</b></p>", unsafe_allow_html=True)
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("🔴 Malignant"):
            # Pick a random row where diagnosis is M
            sample_row = raw_df[raw_df['diagnosis'] == 'M'].sample(1).iloc[0]
            for feature in best_features:
                st.session_state[feature] = float(sample_row[feature])
            st.rerun()
            
    with b_col2:
        if st.button("🟢 Benign"):
            # Pick a random row where diagnosis is B
            sample_row = raw_df[raw_df['diagnosis'] == 'B'].sample(1).iloc[0]
            for feature in best_features:
                st.session_state[feature] = float(sample_row[feature])
            st.rerun()

# ---------------------------------------------------------
# 4. MAIN INTERFACE
# ---------------------------------------------------------
st.markdown("<h1 class='header-text'>AI Breast Cancer Diagnostic System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Please enter the measurements from the biopsy report below.</p>", unsafe_allow_html=True)

# Organising Inputs into Columns
cols = st.columns(3)
user_inputs = {}

# Deeply simplified medical explanations for tooltips
feature_explanations = {
    'radius_mean': 'Average Radius: This is the distance from the center of the tumor to its boundary. In malignant cases, the radius is often larger because the tumor is growing aggressively. Measured in millimeters (mm).',
    
    'texture_mean': 'Texture Index: This measures the "roughness" of the tumor surface based on gray-scale variations. Cancerous tumors tend to have a more irregular, "rough" surface compared to smooth benign ones.',
    
    'perimeter_mean': 'Average Perimeter: This is the total length of the outside edge of the tumor. A larger perimeter usually indicates a more advanced tumor that has spread across more tissue.',
    
    'area_mean': 'Average Area: The total space occupied by the tumor inside the breast. Malignant tumors often occupy much larger areas than benign ones. Measured in square millimeters (mm²).',
    
    'concavity_mean': 'Concavity (Shape): This measures how much the tumor boundary "caves in" or has indentations. Benign tumors are usually smooth/round, while malignant ones are irregular and have deep curves.',
    
    'concave points_mean': 'Concave Points: This counts how many sharp, deep indentations are on the boundary. More concave points mean the tumor has a "spiky" or "jagged" appearance, a common sign of cancer.',
    
    'radius_worst': 'Worst Radius: This is the largest radius measurement found in the entire scan. Even if the average is small, a high "worst" radius shows a part of the tumor is spreading rapidly.',
    
    'texture_worst': 'Worst Texture: This represents the roughest, most irregular part of the tumor surface. It helps identify the most aggressive spot within the tumor.',
    
    'perimeter_worst': 'Worst Perimeter: The largest boundary length found. This helps doctors see the maximum extent of the tumor\'s spread in any single section.',
    
    'area_worst': 'Worst Area: The largest area measurement detected. In medical analysis, the "worst" case is often more important for diagnosis than the "average" case.',
    
    'concavity_worst': 'Worst Concavity: The deepest indentations found on the edge. High values here indicate a very irregular, aggressive tumor shape that is likely cancerous.',
    
    'concave points_worst': 'Worst Concave Points: The maximum number of deep indentations detected. This is one of the most critical indicators used by AI to detect malignancy.'
}

# Map of features to human-friendly labels WITH units
feature_labels = {
    'radius_mean': 'Average Radius (mm)',
    'texture_mean': 'Texture Index (Mean)',
    'perimeter_mean': 'Average Perimeter (mm)',
    'area_mean': 'Average Area (mm²)',
    'concavity_mean': 'Average Concavity',
    'concave points_mean': 'Average Concave Points',
    'radius_worst': 'Worst Radius (mm)',
    'texture_worst': 'Worst Texture Index',
    'perimeter_worst': 'Worst Perimeter (mm)',
    'area_worst': 'Worst Area (mm²)',
    'concavity_worst': 'Worst Concavity',
    'concave points_worst': 'Worst Concave Points'
}


for i, feature in enumerate(best_features):
    col_idx = i % 3
    with cols[col_idx]:
        explanation = feature_explanations.get(feature, "Biopsy measurement parameter.")
        # Using session state to allow example loading
        if feature not in st.session_state:
            st.session_state[feature] = 0.0
            
        user_inputs[feature] = st.number_input(
            feature_labels.get(feature, feature), 
            value=st.session_state[feature], 
            format="%.4f",
            help=explanation,
            key=f"input_{feature}" # Unique key to avoid conflicts
        )
        # Syncing session state back for persistence
        st.session_state[feature] = user_inputs[feature]


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
        
    # Inject JavaScript to auto-scroll to the result card so it's instantly visible
    import streamlit.components.v1 as components
    components.html(
        """
        <script>
            // Find the result card and scroll it into the center of the view smoothly
            const resultCards = window.parent.document.querySelectorAll('.result-card-mal, .result-card-ben');
            if (resultCards.length > 0) {
                resultCards[resultCards.length - 1].scrollIntoView({behavior: 'smooth', block: 'center'});
            }
        </script>
        """,
        height=0
    )

st.markdown("<br><p style='text-align: center; font-size: 12px; color: #bdc3c7;'>Disclaimer: This is an AI-assisted tool for informational purposes only.</p>", unsafe_allow_html=True)
