import streamlit as st
import pandas as pd
import joblib
import gdown
import os

# Google Drive File ID (بدل XXXXX بالـ ID بتاعك)
GOOGLE_DRIVE_ID = "https://drive.google.com/uc?id=1BQLQU3RqH-rLErcVSC0kwPTD0e9I7Rks"
MODEL_PATH = "rf_model.pkl"

@st.cache_resource
def load_model():
    # لو الموديل مش موجود في السيرفر نزله من Google Drive
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
        st.write("⬇️ Downloading model from Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False)

    # تحميل الموديل
    return joblib.load(MODEL_PATH)

# تحميل الموديل
model = load_model()

st.title("✈️ Flight Price Prediction App (Random Forest)")

# --- Inputs ---
airline = st.selectbox("Airline", ["AirAsia", "Air India", "GoAir", "IndiGo", "SpiceJet", "Vistara"])
source_city = st.selectbox("Source City", ["Banglore", "Chennai", "Delhi", "Kolkata", "Mumbai", "Hyderabad"])
departure_time = st.selectbox("Departure Time", ["Morning", "Afternoon", "Evening", "Night", "Early Morning", "Late Night"])
stops = st.selectbox("Stops", ["non-stop", "1 stop", "2 stops"])
destination_city = st.selectbox("Destination City", ["Banglore", "Chennai", "Delhi", "Kolkata", "Mumbai", "Hyderabad"])
travel_class = st.selectbox("Class", ["Economy", "Business"])

duration = st.number_input("Duration (hours)", min_value=0.0, step=0.1)
days_left = st.number_input("Days Left Before Flight", min_value=0, step=1)

# --- Predict ---
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        "airline": [airline],
        "source_city": [source_city],
        "departure_time": [departure_time],
        "stops": [stops],
        "destination_city": [destination_city],
        "class": [travel_class],
        "duration": [duration],
        "days_left": [days_left]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Price: ${prediction:,.2f}")
