import streamlit as st
import pandas as pd
import joblib
import os
import gdown  # pip install gdown

MODEL_PATH = "rf_model.pkl"
GOOGLE_DRIVE_URL = "https://drive.google.com/file/d/1BQLQU3RqH-rLErcVSC0kwPTD0e9I7Rks/view?usp=sharing"  # <-- حط هنا ID بتاع الملف من Google Drive

# --- Download model if not exists ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

model = load_model()

# --- Streamlit UI ---
st.title("Flight Price Prediction App (Random Forest)")

airline = st.selectbox("Airline", ["AirAsia", "Air India", "GoAir", "IndiGo", "SpiceJet", "Vistara"])
source_city = st.selectbox("Source City", ["Banglore", "Chennai", "Delhi", "Kolkata", "Mumbai", "Hyderabad"])
departure_time = st.selectbox("Departure Time", ["Morning", "Afternoon", "Evening", "Night", "Early Morning", "Late Night"])
stops = st.selectbox("Stops", ["non-stop", "1 stop", "2 stops"])
destination_city = st.selectbox("Destination City", ["Banglore", "Chennai", "Delhi", "Kolkata", "Mumbai", "Hyderabad"])
travel_class = st.selectbox("Class", ["Economy", "Business"])

duration = st.number_input("Duration (hours)", min_value=0.0, step=0.1)
days_left = st.number_input("Days Left Before Flight", min_value=0, step=1)

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
    usd_price = prediction   # تقريبًا 1 USD = 83 INR
    st.success(f"Predicted Price: ${usd_price:,.2f} USD")
