# weather_streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="🌦 Weather Data App", layout="wide")
st.title("🌦 Weather Data Streamlit App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your Weather CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    st.subheader("✅ Dataset Preview")
    st.dataframe(df.head(10))
    
    st.subheader("📊 Dataset Info")
    buffer = df.info(buf=None)
    st.text(buffer)
    
    st.subheader("📌 Basic Statistics")
    st.write(df.describe())
    
    # Select column to plot
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if numeric_columns:
        column_to_plot = st.selectbox("Select numeric column to plot", numeric_columns)
        
        st.subheader(f"📈 Plot of {column_to_plot}")
        fig, ax = plt.subplots()
        df[column_to_plot].plot(kind='line', ax=ax, color='skyblue', marker='o')
        ax.set_ylabel(column_to_plot)
        ax.set_xlabel("Index")
        ax.set_title(f"{column_to_plot} over rows")
        st.pyplot(fig)
    else:
        st.warning(" No numeric columns found to plot.")
else:
    st.info("Please upload a CSV file to get started.")
