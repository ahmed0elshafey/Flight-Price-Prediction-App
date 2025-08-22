import streamlit as st
import pandas as pd
import joblib

model = joblib.load("rf_model.pkl")

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
    st.success(f"Predicted Price: {prediction:,.2f} $")
