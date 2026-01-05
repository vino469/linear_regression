import streamlit as st
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Page config
# =========================
st.set_page_config(page_title="Linear Regression App", layout="centered")

st.title("🌤 Linear Regression using Saved Model")

# =========================
# Load model (safe method)
# =========================
MODEL_PATH = "linear_regression_model.pkl"

model = None

# Option 1: Load model from repo
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded from repository")

# Option 2: Upload model manually (VERY IMPORTANT FIX)
else:
    st.warning("⚠ Model file not found in repository")
    uploaded_model = st.file_uploader(
        "Upload linear_regression_model.pkl",
        type=["pkl"]
    )

    if uploaded_model is not None:
        model = joblib.load(uploaded_model)
        st.success("✅ Model uploaded and loaded successfully")

# Stop app if model is still not available
if model is None:
    st.info("⬆ Please upload the model file to continue")
    st.stop()

# =========================
# Prediction Section
# =========================
st.subheader("🔢 Predict Temperature")

humidity = st.number_input(
    "Enter Humidity (%)",
    min_value=0,
    max_value=100,
    value=50
)

if st.button("Predict Temperature"):
    prediction = model.predict([[humidity]])
    st.success(f"🌡 Predicted Temperature: **{prediction[0]:.2f} °C**")

# =========================
# Regression Graph
# =========================
st.subheader("📈 Regression Line")

humidity_values = np.arange(0, 101, 5).reshape(-1, 1)
predicted_temp = model.predict(humidity_values)

fig, ax = plt.subplots()
ax.plot(humidity_values, predicted_temp, marker="o")
ax.set_xlabel("Humidity (%)")
ax.set_ylabel("Predicted Temperature (°C)")
ax.set_title("Humidity vs Temperature")
ax.grid(True)

st.pyplot(fig)
