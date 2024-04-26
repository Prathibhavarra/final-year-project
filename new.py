import streamlit as st
import requests
import pandas as pd
import numpy as np
import pickle

# Function to fetch real-time data from ThingSpeak
def fetch_data_from_thingspeak():
    # Make HTTP GET request to fetch data from ThingSpeak API
    response = requests.get('https://api.thingspeak.com/channels/2469592/feeds.json?results=1')
    if response.status_code == 200:
        data = response.json()
        # Extract relevant information from the response
        # For example, if your data includes temperature, pulse, ecg, age, and gender
        temperature = float(data['feeds'][0]['field1'])
        pulse = float(data['feeds'][0]['field2'])
        ecg = float(data['feeds'][0]['field3'])
        age = float(data['feeds'][0]['field4'])
        gender = data['feeds'][0]['field5']  # Assuming gender is a string field
        return temperature, pulse, ecg, age, gender
    else:
        st.error("Failed to fetch data from ThingSpeak")

# Function to predict stress level for real-time data
def predict_stress_level(temperature, pulse, ecg, age, gender):
    # Load the trained model and scaler
    with open('model.pkl', 'rb') as f:
        svm_classifier = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Encode gender using one-hot encoding
    gender_encoded = 1 if gender.lower() == 'male' else 0
    
    # Scale the input features
    new_patient_data = np.array([[temperature, pulse, ecg, age, 1 - gender_encoded, gender_encoded]])
    new_patient_data_scaled = scaler.transform(new_patient_data)
    
    # Predict the stress level
    predicted_stress = svm_classifier.predict(new_patient_data_scaled)
    
    return predicted_stress[0]

# Streamlit UI
st.title("Real-Time Stress Prediction")

# Fetch real-time data from ThingSpeak
temperature, pulse, ecg, age, gender = fetch_data_from_thingspeak()

# Display fetched data
st.write("Fetched Real-Time Data:")
st.write("Temperature:", temperature)
st.write("Pulse:", pulse)
st.write("ECG:", ecg)
st.write("Age:", age)
st.write("Gender:", gender)

# Predict stress level
predicted_stress = predict_stress_level(temperature, pulse, ecg, age, gender)
st.write("Predicted Stress Level:", predicted_stress)