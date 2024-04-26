import streamlit as st
import numpy as np
import pickle
import requests

# Function to fetch data from the API and calculate average values
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
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException as e:
        st.error(f'Error connecting to ThingSpeak API: {e}')
        return None, None, None

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
        st.error(f'Error retrieving data from ThingSpeak API: Status code {response.status_code}')
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

# Function to load the trained models and scaler
@st.cache(allow_output_mutation=True)
def load_models():
    # Load the trained models for stress, heatstroke, and arrhythmia prediction
    with open('svm_stress_model.pkl', 'rb') as f:
        svm_stress_model = pickle.load(f)
    
    with open('svm_heatstroke_model.pkl', 'rb') as f:
        svm_heatstroke_model = pickle.load(f)

    with open('svm_arrhythmia_model.pkl', 'rb') as f:
        svm_arrhythmia_model = pickle.load(f)

    # Load the scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return svm_stress_model, svm_heatstroke_model, svm_arrhythmia_model, scaler

# Function to predict stress level
def predict_stress_level(average_temperature, average_pulse, average_ecg, age, gender):
    # Load the trained stress model
    svm_stress_model = load_models()[0]
    scaler = load_models()[3]
    
    # Encode gender
    gender_encoded = 1 if gender.lower() == 'male' else 0
    
    # Scale the input features
    new_patient_data = np.array([[average_temperature, average_pulse, average_ecg, age, 1 - gender_encoded, gender_encoded]])
    new_patient_data_scaled = scaler.transform(new_patient_data)
    
    # Predict stress level
    predicted_stress = svm_stress_model.predict(new_patient_data_scaled)
    return predicted_stress[0]

# Function to predict heatstroke
def predict_heatstroke(average_temperature, average_pulse, average_ecg, age, gender):
    # Load the trained heatstroke model
    svm_heatstroke_model = load_models()[1]
    scaler = load_models()[3]
    
    # Encode gender
    gender_encoded = 1 if gender.lower() == 'male' else 0
    
    # Scale the input features
    new_patient_data = np.array([[average_temperature, average_pulse, average_ecg, age, 1 - gender_encoded, gender_encoded]])
    new_patient_data_scaled = scaler.transform(new_patient_data)
    
    # Predict heatstroke
    predicted_heatstroke = svm_heatstroke_model.predict(new_patient_data_scaled)
    return predicted_heatstroke[0]

# Function to predict arrhythmia
def predict_arrhythmia(average_temperature, average_pulse, average_ecg, age, gender):
    # Load the trained arrhythmia model
    svm_arrhythmia_model = load_models()[2]
    scaler = load_models()[3]
    
    # Encode gender
    gender_encoded = 1 if gender.lower() == 'male' else 0
    
    # Scale the input features
    new_patient_data = np.array([[average_temperature, average_pulse, average_ecg, age, 1 - gender_encoded, gender_encoded]])
    new_patient_data_scaled = scaler.transform(new_patient_data)
    
    # Predict arrhythmia
    predicted_arrhythmia = svm_arrhythmia_model.predict(new_patient_data_scaled)
    return predicted_arrhythmia[0]

# Streamlit web app layout
st.title('Health Status Prediction')

# Input widgets
age = st.slider('Age', min_value=18, max_value=100, step=1)
gender = st.radio('Gender', ('Male', 'Female'))

# Predict health status
if st.button('Predict Health Status'):
    # Fetch data from the API and calculate average values
    average_temperature, average_pulse, average_ecg = fetch_and_calculate_averages()
    
    # Check if average values are available
    if average_temperature is not None and average_pulse is not None and average_ecg is not None:
        # Predict stress level using fetched data
        predicted_stress = predict_stress_level(average_temperature, average_pulse, average_ecg, age, gender)
        
        # Predict heatstroke using fetched data
        predicted_heatstroke = predict_heatstroke(average_temperature, average_pulse, average_ecg, age, gender)
        
        # Predict arrhythmia using fetched data
        predicted_arrhythmia = predict_arrhythmia(average_temperature, average_pulse, average_ecg, age, gender)

        # Display the average values of temperature, pulse, and ECG
        st.write("Average Temperature:", average_temperature)
        st.write("Average Pulse:", average_pulse)
        st.write("Average ECG:", average_ecg)
        
        # Display the predicted health status
        st.write(f'Predicted Stress Level: {predicted_stress}')
        st.write(f'Predicted Heatstroke: {predicted_heatstroke}')
        st.write(f'Predicted Arrhythmia: {predicted_arrhythmia}')