# phq9_streamlit_app.py
# ---------------------------------------
# Streamlit app to load a pickled StandardScaler + LinearRegression pipeline
# and predict PHQ-9 total score from 14 IV inputs. If predicted score < 10
# → "No depression"; otherwise → "Depression".
# ---------------------------------------

import pickle
from pathlib import Path

import numpy as np
import streamlit as st

# -------------------------
# Configuration & Constants
# -------------------------
MODEL_PATH = Path("lin_pipe.pkl")  # Adjust if the file lives elsewhere

FEATURES = [
    "Qi-stagnation",
    "Blood-stasis",
    "Qi-deficiency",
    "Yang-deficiency",
    "Yin-deficiency",
    "Phlegm-dampness",
    "Damp-heat",
    "Inherited special",
    "Balanced",
    "Dysfunctional attitude",
    "Stress",
    "Anxiety",
    "Perceived Stress",
    "Self-esteem",
]

THRESHOLD = 10  # PHQ-9 score cut-off for depression

# -------------------------
# Helpers
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    """Load pickled scikit-learn pipeline."""
    if not path.exists():
        st.error(f"Pickle file not found: {path.resolve()}")
        st.stop()
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


# -------------------------
# Main UI
# -------------------------
st.set_page_config(page_title="PHQ-9 Depression Predictor", page_icon="🧠")
st.title("🧠 PHQ-9 Depression Predictor")

st.markdown(
    """
    Enter the measured values for each Traditional Chinese Medicine (TCM) pattern and
    psychosocial scale below. Click **Predict** to estimate the patient's PHQ-9 total
    score and see the corresponding depression category.
    
    *Scores below **10** → **No Depression**  
    Scores **10 or above** → **Depression***
    """,
    unsafe_allow_html=True,
)

# Load model once
model = load_model(MODEL_PATH)

# Collect user inputs in three columns
inputs = {}
st.subheader("Input the Independent Variables (IVs)")
cols = st.columns(3)
for idx, feature in enumerate(FEATURES):
    with cols[idx % 3]:
        inputs[feature] = st.number_input(
            label=feature,
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
            key=feature,
            help="Enter the raw score for this scale",
        )

# Prediction button
if st.button("Predict", type="primary"):
    X = np.array([inputs[f] for f in FEATURES]).reshape(1, -1)
    try:
        y_pred = float(model.predict(X)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("Prediction Result")
    st.metric(label="Predicted PHQ-9 Score", value=f"{y_pred:.2f}")

    if y_pred < THRESHOLD:
        st.success("No Depression (score < 10)")
    else:
        st.error("Depression (score ≥ 10)")

    st.caption(
        "*Disclaimer: This tool is for educational purposes only and does not replace professional mental-health assessment.*"
    )
