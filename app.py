# app_ui.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="🌱 Smart Crop Recommender", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("crop_recommendation_model.pkl")

model = load_model()
feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Sidebar for inputs
st.sidebar.header("🧪 Input Soil & Weather Conditions")

N = st.sidebar.slider("Nitrogen (N)", 0, 200, 90)
P = st.sidebar.slider("Phosphorus (P)", 0, 200, 42)
K = st.sidebar.slider("Potassium (K)", 0, 200, 43)
temperature = st.sidebar.slider("Temperature (°C)", -10.0, 60.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 80.0)
ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 500.0, 200.0)

# Create input DataFrame
input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_names)

st.title("🌾 Smart Crop Recommendation System")
st.write("This app suggests the **best crop** to grow based on your soil and climate conditions.")

if st.button("🚀 Recommend Crop"):
    prediction = model.predict(input_data)
    st.success(f"✅ Best Crop to Grow: **{prediction[0]}**")

    # Show top 3 predictions if available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        crops = model.classes_
        crop_probs = sorted(zip(crops, probs), key=lambda x: x[1], reverse=True)[:3]

        st.subheader("🔝 Top 3 Crop Predictions")
        for crop, prob in crop_probs:
            st.progress(float(prob))
            st.write(f"🌱 {crop}: **{prob:.2%}**")
