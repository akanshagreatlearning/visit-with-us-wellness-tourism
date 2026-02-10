import os
import pandas as pd
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Package Predictor", layout="centered")

# ----------------------------
# Load model from HF Model Hub
# ----------------------------
@st.cache_resource
def load_model():
    model_repo_id = os.getenv("HF_MODEL_REPO_ID", "akanshasalampuria/visit-with-us-wellness-model")
    model_path = hf_hub_download(repo_id=model_repo_id, repo_type="model", filename="model.joblib")
    return joblib.load(model_path)

model = load_model()

# Try to detect expected feature columns from the trained pipeline/model
EXPECTED_COLS = None
try:
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        EXPECTED_COLS = list(cols)
except Exception:
    EXPECTED_COLS = None

st.title("Visit with Us – Wellness Tourism Package Predictor")
st.write("Upload a 1-row CSV (features only) to predict whether the customer will purchase the Wellness Tourism Package.")

with st.expander("Show expected input columns (recommended)"):
    if EXPECTED_COLS is None:
        st.info("Could not detect expected columns from the model object. Use a CSV matching the training feature set.")
    else:
        st.write(EXPECTED_COLS)

# -------------------------------------------
# CSV Upload (ONLY)
# -------------------------------------------
st.subheader("Upload 1-row CSV")
uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

input_df = None
is_valid = False
missing_cols = []
extra_cols = []

if uploaded is not None:
    df = pd.read_csv(uploaded)

    # Drop accidental index column if present (common issue)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if df.shape[0] != 1:
        st.error("Please upload a CSV with exactly one row (one customer).")
    else:
        # Validate / align columns
        if EXPECTED_COLS is not None:
            missing_cols = [c for c in EXPECTED_COLS if c not in df.columns]
            extra_cols = [c for c in df.columns if c not in EXPECTED_COLS]

            if missing_cols:
                st.error("CSV validation failed: missing required columns.")
                st.write("**Missing columns:**", missing_cols)
                if extra_cols:
                    st.write("**Extra columns (not required):**", extra_cols)
                is_valid = False
            else:
                if extra_cols:
                    st.warning(f"CSV contains extra columns not used by the model: {extra_cols}")

                # Align order and fill any missing (should be none here) with 0
                df = df.reindex(columns=EXPECTED_COLS, fill_value=0)
                input_df = df
                is_valid = True
        else:
            # If we cannot detect expected cols, allow prediction attempt
            input_df = df
            is_valid = True

        st.success("CSV loaded successfully.")
        st.dataframe(df)

# -------------------------------------------
# Predict
# -------------------------------------------
st.subheader("Prediction")
if st.button("Predict Now"):
    if input_df is None:
        st.warning("Please upload a valid 1-row CSV first.")
    elif not is_valid:
        st.warning("Your CSV is not valid for prediction. Please fix the validation errors above.")
    else:
        try:
            pred = model.predict(input_df)[0]
            st.success(f"Prediction (0 = No, 1 = Yes): **{int(pred)}**")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0][1]
                st.info(f"Purchase probability: **{proba:.2%}**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.caption("Tip: Ensure the uploaded CSV column names and data types match the training data.")
