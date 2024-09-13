import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('heart_failure_model.joblib')

# Define the input form
st.title("Heart Disease Prediction App")
st.write("Enter patient details to predict the possibility of heart disease")

# Create the input fields arranged according to the specified order
col1, col2, col3 = st.columns(3)

with col1:
    sex = st.selectbox("Sex", ["Male", "Female"])
    chestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    restingBP = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fastingBS = st.number_input("Fasting Blood Sugar (mg/dl)", min_value=0, max_value=600, value=120)

with col2:
    restingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    maxHR = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=100)
    exerciseAngina = st.selectbox("Exercise Angina", ["Yes", "No"])

with col3:
    oldpeak = st.number_input("Old Peak", min_value=0.0, max_value=10.0, value=1.0)
    st_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Prepare the input data for prediction
input_data = pd.DataFrame([[sex, chestPainType, restingBP, cholesterol, fastingBS, restingECG, maxHR, exerciseAngina, oldpeak, st_Slope]],
                          columns=['Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'])

# Encode the categorical variables
def preprocess_input(data):
    data['Sex'] = 1 if data['Sex'][0] == 'Male' else 0
    data['ChestPainType'] = {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3}[data['ChestPainType'][0]]
    data['RestingECG'] = {'Normal': 0, 'ST': 1, 'LVH': 2}[data['RestingECG'][0]]
    data['ExerciseAngina'] = 1 if data['ExerciseAngina'][0] == 'Yes' else 0
    data['ST_Slope'] = {'Up': 0, 'Flat': 1, 'Down': 2}[data['ST_Slope'][0]]
    
    # Convert fastingBS to 1 if > 120, else 0
    data['FastingBS'] = 1 if data['FastingBS'][0] > 120 else 0
    
    return data

# Preprocess input data
input_data = preprocess_input(input_data)

# Predict Button
if st.button('Predict'):
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 0:
            st.write("The model predicts **No Heart Disease**.")
        else:
            st.write("The model predicts **Heart Disease**.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
