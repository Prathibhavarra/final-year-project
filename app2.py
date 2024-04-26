import streamlit as st
import requests
import numpy as np
import pickle

# Function to fetch data from the API and calculate average values
@st.cache_data()
def fetch_and_calculate_averages():
    # ThingSpeak Channel ID and Read API Key
    channel_id = '2469592'
    read_api_key = 'F98GZRMX2FT7Y8NV'

    # ThingSpeak API endpoint URL
    url = f'https://api.thingspeak.com/channels/{channel_id}/feeds.json?api_key={read_api_key}'

    # Lists to store valid temperature, pulse, and ECG values
    valid_temperature_values = []
    valid_pulse_values = []
    valid_ecg_values = []

    # Make GET request to ThingSpeak API
    response = requests.get(url)

    # Check if request was successful
    if response.status_code == 200:
        # Parse JSON response
        data = response.json()
        
        # Check if the 'feeds' key exists and contains data
        if 'feeds' in data and data['feeds']:
            # Extract relevant data (temperature, pulse, ECG)
            for entry in data['feeds']:
                temperature = entry.get('field1')
                pulse = entry.get('field2')
                ecg = entry.get('field3')
                
                # Check if temperature value is valid and append it to the list
                if temperature is not None and temperature != '' and float(temperature) > 0:
                    valid_temperature_values.append(float(temperature))
                # Check if pulse value is valid and append it to the list
                if pulse is not None and pulse != '' and float(pulse) > 0:
                    valid_pulse_values.append(float(pulse))
                # Check if ECG value is valid and append it to the list
                if ecg is not None and ecg != '' and float(ecg) > 0:
                    valid_ecg_values.append(float(ecg))
        else:
            st.error('No data available in response.')
            return None, None, None
    else:
        st.error('Error retrieving data from ThingSpeak')
        return None, None, None

    # Calculate the average of the valid values
    def calculate_average(values):
        if values:
            return sum(values) / len(values)
        else:
            return None

    # Calculate averages for each parameter
    average_temperature = calculate_average(valid_temperature_values)
    average_pulse = calculate_average(valid_pulse_values)
    average_ecg = calculate_average(valid_ecg_values)

    return average_temperature, average_pulse, average_ecg

# Function to load the trained model
@st.cache_data()
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Function to predict stress levels
def predict_stress_level(average_temperature, average_pulse, average_ecg, age, gender):
    # Load the trained model
    svm_classifier = load_model()
    
    # Encode gender
    gender_encoded = 1 if gender.lower() == 'male' else 0
    
    # Create input array for prediction
    new_patient_data = np.array([[average_temperature, average_pulse, average_ecg, age, 1 - gender_encoded, gender_encoded]])
    
    # Predict stress level
    predicted_stress = svm_classifier.predict(new_patient_data)
    return predicted_stress[0]

# Streamlit web app layout
st.title('Stress Level Prediction')

# Input widgets
age = st.slider('Age', min_value=18, max_value=100, step=1)
gender = st.radio('Gender', ('Male', 'Female'))

# Predict stress level
if st.button('Predict Stress Level'):
    # Fetch data from the API and calculate average values
    average_temperature, average_pulse, average_ecg = fetch_and_calculate_averages()
    
    # Check if average values are available
    if average_temperature is not None and average_pulse is not None and average_ecg is not None:
        # Predict stress level using fetched data
        predicted_stress = predict_stress_level(average_temperature, average_pulse, average_ecg, age, gender)
        
        # Display the average values of temperature, pulse, and ECG
        st.write("Average Temperature:", average_temperature)
        st.write("Average Pulse:", average_pulse)
        st.write("Average ECG:", average_ecg)
        
        # Display the predicted stress level
        st.write(f'Predicted Stress Level: {predicted_stress}')
