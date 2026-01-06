import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression

# -----------------------------
# Try importing plotly safely
# -----------------------------
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ModuleNotFoundError:
    PLOTLY_AVAILABLE = False

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Stock Price Predictor",
    layout="wide",
    page_icon="📈"
)

# -----------------------------
# Title
# -----------------------------
st.title("📈 Stock Price Predictor")
st.write("Predict stock prices based on trading volume using **Linear Regression**")

# -----------------------------
# Sidebar input
# -----------------------------
st.sidebar.header("📊 Input")
volume = st.sidebar.number_input(
    "Enter Trading Volume (millions)",
    min_value=0.1,
    max_value=10.0,
    value=2.5,
    step=0.1
)

# -----------------------------
# Train model
# -----------------------------
@st.cache_resource
def train_model():
    data = {
        "volume": [1.79, 2.03, 2.5, 3.2],
        "price": [114.3, 114.2, 150.0, 220.0]
    }
    df = pd.DataFrame(data)

    model = LinearRegression()
    model.fit(df[["volume"]], df["price"])

    return model, df

model, df = train_model()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Price"):
    pred_price = model.predict([[volume]])[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Price", f"${pred_price:,.2f}")
    col2.metric("Volume", f"{volume:.2f} M")
    col3.metric("Slope", f"{model.coef_[0]:.2f}")

    st.success(
        f"For **{volume:.2f}M volume**, predicted stock price is **${pred_price:,.2f}**"
    )

# -----------------------------
# Visualization
# -----------------------------
st.header("📈 Visualization")

col1, col2 = st.columns(2)

# ---- Matplotlib (ALWAYS WORKS)
with col1:
    fig, ax = plt.subplots()

    ax.scatter(df["volume"], df["price"], label="Training Data")
    x = np.linspace(df["volume"].min(), df["volume"].max(), 100)
    y = model.predict(x.reshape(-1, 1))
    ax.plot(x, y, label="Regression Line")

    ax.scatter(volume, model.predict([[volume]])[0], color="red", marker="*",
               s=150, label="Prediction")

    ax.set_xlabel("Volume (Millions)")
    ax.set_ylabel("Price ($)")
    ax.legend()
    st.pyplot(fig)

# ---- Plotly (ONLY if available)
with col2:
    if PLOTLY_AVAILABLE:
        fig = px.scatter(
            df,
            x="volume",
            y="price",
            title="Training Data (Interactive)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Plotly not installed. Interactive chart disabled.")

# -----------------------------
# Batch Prediction
# -----------------------------
st.header("📋 Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV with 'volume' column", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    if "volume" in test_df.columns:
        test_df["predicted_price"] = model.predict(test_df[["volume"]])
        st.dataframe(test_df)

        csv = test_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )
    else:
        st.error("CSV must contain 'volume' column")

# -----------------------------
# Save / Load model
# -----------------------------
if st.button("💾 Save Model"):
    joblib.dump(model, "stock_model.joblib")
    st.success("Model saved successfully")

if st.button("📥 Load Model"):
    try:
        loaded_model = joblib.load("stock_model.joblib")
        st.success("Model loaded")
        st.write("Coefficient:", loaded_model.coef_[0])
    except:
        st.warning("No saved model found")

st.markdown("---")
st.caption("Built with Streamlit & Scikit-Learn")

