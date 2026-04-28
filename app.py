import streamlit as st
import numpy as np
import os

# -------------------------
# PAGE CONFIGURATION
# -------------------------
st.set_page_config(
    page_title="Cancer Risk Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# CUSTOM CSS
# -------------------------
def local_css():
    st.markdown("""
    <style>
    /* Base Theme overrides */
    [data-testid="stAppViewContainer"] {
        background-color: #0f172a;
        background-image: radial-gradient(circle at 50% 0%, #1e293b 0%, #0f172a 100%);
        color: #f8fafc;
    }

    [data-testid="stHeader"] {
        background-color: transparent;
    }

    /* Custom Typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.05em;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Glassmorphism for Form */
    [data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5) !important;
        backdrop-filter: blur(12px) !important;
        padding: 2rem !important;
    }

    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 1px solid #334155;
    }

    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
    }

    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }

    /* Styled Button */
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        width: 100% !important;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.5) !important;
    }

    /* Result cards */
    .result-container {
        padding: 2.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        margin-top: 1rem;
    }

    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px) scale(0.95); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }

    .high-risk {
        background: linear-gradient(145deg, rgba(239, 68, 68, 0.1), rgba(153, 27, 27, 0.05));
        border: 1px solid rgba(239, 68, 68, 0.2);
        box-shadow: 0 20px 40px -10px rgba(239, 68, 68, 0.15);
    }

    .low-risk {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.1), rgba(6, 78, 59, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.2);
        box-shadow: 0 20px 40px -10px rgba(16, 185, 129, 0.15);
    }
    
    .metric-value {
        font-size: 4rem; 
        font-weight: 800; 
        line-height: 1.1;
        margin-bottom: 0.5rem;
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.1);
        margin: 1.5rem 0;
    }
    
    /* Slider Customization */
    .stSlider > div > div > div > div {
        background: #3b82f6 !important;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# -------------------------
# LOAD MODEL COMPONENTS
# -------------------------

try:
    mean_vals = np.load("mean.npy")
    std_vals = np.load("std.npy")
    feature_bounds = np.load("feature_bounds.npy") # [min, max, mean]
except FileNotFoundError:
    st.error("Model files not found. Please run `python explore.py` first to train the model and generate files.")
    st.stop()

# Import your model class
from logistic_model import LogisticRegressionScratch

# Recreate model and load weights
model = LogisticRegressionScratch()

try:
    model.weights = np.load("weights.npy")
    model.bias = np.load("bias.npy")
except FileNotFoundError:
    st.error("Model weights not found. Please run `python explore.py` first to train the model.")
    st.stop()

# -------------------------
# HEADER & BANNER
# -------------------------

# Try to load the generated banner image if it exists
banner_path = "dna_medical_header.png"
if os.path.exists(banner_path):
    st.image(banner_path, use_column_width=True)

st.markdown("<h1 style='text-align: center; margin-top: -1rem;'>🧬 Cancer Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>Advanced Machine Learning Analysis of Tumor Cytological Features</p>", unsafe_allow_html=True)

# -------------------------
# UI LAYOUT
# -------------------------

features = [
    "concave points_worst", "perimeter_worst", "radius_worst", "perimeter_mean", 
    "area_worst", "radius_mean", "area_mean", "concave points_mean", 
    "concavity_worst", "concavity_mean", "fractal_dimension_worst", "fractal_dimension_mean"
]

# Calculate bounds and metadata for each feature
inputs_dict = {}
for i, f in enumerate(features):
    min_val = float(feature_bounds[0][i])
    max_val = float(feature_bounds[1][i])
    m_val = float(feature_bounds[2][i])
    
    pad = (max_val - min_val) * 0.1
    min_val = max(0.0, min_val - pad) # Features generally cannot be negative
    max_val = max_val + pad
    step = (max_val - min_val) / 100.0 if (max_val - min_val) > 0 else 0.01
    
    inputs_dict[f] = {
        "min": min_val,
        "max": max_val,
        "mean": m_val,
        "step": step,
        "index": i,
        "value": m_val # Default value
    }

col1, col2 = st.columns([1.8, 1.2], gap="large")

with col1:
    with st.form("prediction_form"):
        st.markdown("<h3 style='color: #f8fafc; margin-bottom: 1rem;'>Tumor Features Input</h3>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📊 Mean Measurements", "⚠️ Worst Measurements"])
        
        mean_features = [f for f in features if "mean" in f]
        worst_features = [f for f in features if "worst" in f]
        
        with tab1:
            m_cols = st.columns(2)
            for i, f in enumerate(mean_features):
                col = m_cols[i % 2]
                with col:
                    label = f.replace("_", " ").title()
                    val = st.slider(label, 
                                    min_value=inputs_dict[f]["min"], 
                                    max_value=inputs_dict[f]["max"], 
                                    value=inputs_dict[f]["mean"], 
                                    step=inputs_dict[f]["step"])
                    inputs_dict[f]["value"] = val
                    
        with tab2:
            w_cols = st.columns(2)
            for i, f in enumerate(worst_features):
                col = w_cols[i % 2]
                with col:
                    label = f.replace("_", " ").title()
                    val = st.slider(label, 
                                    min_value=inputs_dict[f]["min"], 
                                    max_value=inputs_dict[f]["max"], 
                                    value=inputs_dict[f]["mean"], 
                                    step=inputs_dict[f]["step"])
                    inputs_dict[f]["value"] = val

        submit_button = st.form_submit_button("Analyze Risk Profile")

with col2:
    st.markdown("<h3 style='color: #f8fafc; margin-bottom: 1rem;'>Analysis Results</h3>", unsafe_allow_html=True)
    
    if submit_button:
        # Build input_data array in exact original order
        input_data = np.zeros(len(features))
        for f, data in inputs_dict.items():
            input_data[data["index"]] = data["value"]
            
        input_data = np.array([input_data])
        
        # Normalize
        input_data_norm = (input_data - mean_vals) / std_vals
        
        # Predict
        probs = model.predict_proba(input_data_norm)
        prob = probs[0]
        pred = 1 if prob > 0.5 else 0
        
        if pred == 1:
            st.markdown(f'''
            <div class="result-container high-risk">
                <h2 style="color: #ef4444; margin-bottom: 1rem;">High Risk Profile</h2>
                <div class="metric-value" style="color: #ef4444;">{prob*100:.1f}%</div>
                <p style="color: #cbd5e1; font-size: 1.1rem; font-weight: 500;">Likelihood of Malignancy (M)</p>
                <hr>
                <p style="color: #fca5a5; font-weight: 600; font-size: 1.1rem;">⚠️ Immediate medical consultation recommended.</p>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 1rem;">This model predicts based on cellular features and should not replace professional medical diagnosis.</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="result-container low-risk">
                <h2 style="color: #10b981; margin-bottom: 1rem;">Low Risk Profile</h2>
                <div class="metric-value" style="color: #10b981;">{prob*100:.1f}%</div>
                <p style="color: #cbd5e1; font-size: 1.1rem; font-weight: 500;">Likelihood of Malignancy (M)</p>
                <hr>
                <p style="color: #6ee7b7; font-weight: 600; font-size: 1.1rem;">✅ Likely Benign Tumor (B)</p>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 1rem;">Continue with routine check-ups as advised by your healthcare provider.</p>
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div style="background: rgba(30, 41, 59, 0.3); border: 1px dashed rgba(148, 163, 184, 0.3); border-radius: 16px; padding: 3rem 2rem; text-align: center; margin-top: 1rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔬</div>
            <h4 style="color: #94a3b8; margin-bottom: 0.5rem;">Awaiting Analysis</h4>
            <p style="color: #64748b; font-size: 0.95rem;">Adjust the tumor cytological features in the panel and click <b>Analyze Risk Profile</b> to view the prediction.</p>
        </div>
        ''', unsafe_allow_html=True)
