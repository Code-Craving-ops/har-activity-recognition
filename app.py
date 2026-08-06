# ============================================================
# STEP 3: STREAMLIT DEMO APP
# RUN THIS ON: your own laptop first (to test), then deploy free
# FILES NEEDED IN SAME FOLDER: har_model.pkl, scaler.pkl, label_encoder.pkl
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Human Activity Recognition", page_icon="🏃", layout="centered")

st.title("🏃 Human Activity Recognition")
st.write(
    "Predicts one of 6 activities — Walking, Walking Upstairs, Walking Downstairs, "
    "Sitting, Standing, Laying — from smartphone accelerometer/gyroscope features."
)

# ---- Load trained model + preprocessing objects (cached so it loads only once) ----
@st.cache_resource
def load_artifacts():
    model = joblib.load("har_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_artifacts()

st.divider()
st.subheader("Input sensor features")

option = st.radio("How do you want to provide input?", ["Upload a CSV row", "Use a sample test row"])

if option == "Upload a CSV row":
    uploaded_file = st.file_uploader(
        "Upload a CSV with 561 feature columns (same format as UCI HAR X_test.txt)", type=["csv"]
    )
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file, header=None)
        st.write("Preview of uploaded data:", input_df.head())

        if st.button("Predict Activity"):
            X_input = scaler.transform(input_df.values)
            pred = model.predict(X_input)
            activity = label_encoder.inverse_transform(pred)
            st.success(f"Predicted Activity: **{activity[0]}**")

else:
    st.info(
        "Demo mode: generates a random plausible feature row so you can show the "
        "app working live without needing a real CSV on hand (e.g., in an interview)."
    )
    if st.button("Generate & Predict Sample"):
        # NOTE: replace this with a real row from X_test.txt for an accurate demo.
        # Random values here are ONLY for showing the app mechanics work.
        dummy_input = np.random.uniform(-1, 1, size=(1, 561))
        X_input = scaler.transform(dummy_input)
        pred = model.predict(X_input)
        activity = label_encoder.inverse_transform(pred)
        st.success(f"Predicted Activity: **{activity[0]}**")
        st.caption("(Demo used random synthetic input — upload real sensor data above for genuine predictions.)")

st.divider()
st.caption("Model: SVM (RBF) trained on UCI HAR Dataset | Built by [Your Name]")
