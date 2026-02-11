import os
import time
import pandas as pd
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Package Predictor", layout="centered")

st.title("Visit with Us – Wellness Tourism Package Predictor")
st.write("Upload a 1-row CSV (features only) to predict purchase likelihood.")

MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "akanshasalampuria/visit-with-us-wellness-model")
MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "model.joblib")  

# ----------------------------
# Safe model loader (retry + cached)
# ----------------------------
@st.cache_resource
def load_model_with_retry(repo_id: str, filename: str, retries: int = 5, wait_sec: int = 3):
    last_err = None
    for _ in range(retries):
        try:
            path = hf_hub_download(repo_id=repo_id, repo_type="model", filename=filename)
            return joblib.load(path)
        except Exception as e:
            last_err = e
            time.sleep(wait_sec)
    raise last_err

# Auto-load at startup
st.subheader("Model Status")
with st.spinner("Loading model..."):
    try:
        model = load_model_with_retry(MODEL_REPO_ID, MODEL_FILENAME)
        st.success("Model loaded and ready.")
    except Exception as e:
        st.error("Could not download the model from Hugging Face Hub right now.")
        st.caption("This may be a temporary HF network/DNS issue. Please refresh and try again.")
        st.exception(e)
        st.stop()

# ----------------------------
# Detect expected columns robustly
# ----------------------------
EXPECTED_COLS = getattr(model, "feature_names_in_", None)
if EXPECTED_COLS is not None:
    EXPECTED_COLS = list(EXPECTED_COLS)

# If pipeline has a named preprocess step with feature_names_in_
if EXPECTED_COLS is None and hasattr(model, "named_steps"):
    pre = model.named_steps.get("preprocess", None)
    if pre is not None:
        pre_cols = getattr(pre, "feature_names_in_", None)
        if pre_cols is not None:
            EXPECTED_COLS = list(pre_cols)

with st.expander("Show expected input columns (recommended)"):
    if EXPECTED_COLS is None:
        st.info("Could not detect expected columns from the model. Use a CSV matching training features.")
    else:
        st.write(EXPECTED_COLS)

# ----------------------------
# Provide template CSV download
# ----------------------------
st.subheader("Sample Input Template")
if EXPECTED_COLS is None:
    st.warning("Template not available because expected columns couldn't be detected.")
else:
    # Create a 1-row dummy template
    example_row = {}
    for c in EXPECTED_COLS:
        # safe defaults
        if c in ["Unnamed: 0", "Unnamed:0"]:
            example_row[c] = 0
        else:
            example_row[c] = ""

    template_df = pd.DataFrame([example_row])

    st.download_button(
        label="Download dummy CSV template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="input_template.csv",
        mime="text/csv",
    )

    with st.expander("Preview template"):
        st.dataframe(template_df)

# ----------------------------
# CSV upload
# ----------------------------
st.subheader("Upload 1-row CSV")
uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

input_df = None
is_valid = False

if uploaded is not None:
    df = pd.read_csv(uploaded)

    if df.shape[0] != 1:
        st.error("Please upload a CSV with exactly one row.")
    else:
        if EXPECTED_COLS is not None:
            # Add missing expected columns (handle Unnamed:0 safely)
            for c in EXPECTED_COLS:
                if c not in df.columns:
                    df[c] = 0 if c in ["Unnamed: 0", "Unnamed:0"] else ""

            # Identify extra columns
            extra_cols = [c for c in df.columns if c not in EXPECTED_COLS]
            if extra_cols:
                st.warning(f"Extra columns ignored: {extra_cols}")

            # Reorder and keep only expected
            df = df[EXPECTED_COLS]
            input_df = df
            is_valid = True
        else:
            input_df = df
            is_valid = True

        st.success("CSV loaded successfully.")
        st.dataframe(input_df)

# ----------------------------
# Predict
# ----------------------------
st.subheader("Prediction")
if st.button("Predict Now"):
    if input_df is None or not is_valid:
        st.warning("Please upload a valid 1-row CSV first.")
    else:
        try:
            pred = model.predict(input_df)[0]
            st.success(f"Prediction (0 = No, 1 = Yes): **{int(pred)}**")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0][1]
                st.info(f"Purchase probability: **{proba:.2%}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
