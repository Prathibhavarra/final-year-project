import streamlit as st
import numpy as np
import pickle

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

svm_classifier = load_model()

# Function to predict stress levels for new patients
def predict_stress_level(temperature, pulse, ecg, age, gender):
    # Calculate the average temperature, pulse, and ECG
    average_temperature = np.mean(temperature)
    average_pulse = np.mean(pulse)
    average_ecg = np.mean(ecg)
    
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
    # Dummy values for temperature, pulse, and ECG (replace with your code to fetch previous values)
    # For now, let's assume average values are 37.5, 80, and 75 respectively
    temperature = [37.5]
    pulse = [80]
    ecg = [75]
    
    predicted_stress = predict_stress_level(temperature, pulse, ecg, age, gender)
    
    # Display the average values of temperature, pulse, and ECG
    st.write(f'Average Temperature: {temperature[0]}')
    st.write(f'Average Pulse: {pulse[0]}')
    st.write(f'Average ECG: {ecg[0]}')
    
    # Display the predicted stress level
    st.write(f'Predicted Stress Level: {predicted_stress}')