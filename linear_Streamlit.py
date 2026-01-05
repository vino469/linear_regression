import streamlit as st
import joblib
import matplotlib.pyplot as plt

st.title("🌤 Linear Regression using Saved Model")

# Load trained model
model = joblib.load("linear_regression_model.pkl")

# Prediction input
st.subheader("Predict Temperature")
humidity = st.number_input("Enter Humidity (%)", min_value=0, max_value=100)

if st.button("Predict"):
    prediction = model.predict([[humidity]])
    st.success(f"Predicted Temperature: {prediction[0]:.2f} °C")

# Regression graph (default range 0–100)
st.subheader("Regression Graph")
humidity_values = [[i] for i in range(0, 101, 5)]
predicted_temp = model.predict(humidity_values)

plt.figure()
plt.plot(range(0, 101, 5), predicted_temp, color='red', marker='o')
plt.xlabel("Humidity (%)")
plt.ylabel("Predicted Temperature")
st.pyplot(plt)
