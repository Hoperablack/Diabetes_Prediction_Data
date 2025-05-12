import pandas as pd
import joblib
import streamlit as st

st.title("Diabetes Prediction App")

model = joblib.load('diabetes_model.joblib')
scaler = joblib.load('scaler.pkl')
columns_to_scale = joblib.load('columns_to_scale.pkl')

pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, step=1)
glucose = st.number_input('Glucose Level', min_value=0)
blood_pressure = st.number_input('Blood Pressure', min_value=0)
skin_thickness = st.number_input('Skin Thickness', min_value=0)
insulin = st.number_input('Insulin Level', min_value=0)
bmi = st.number_input('BMI', min_value=0.0)
diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0)
age = st.number_input('Age', min_value=1)

sex = st.selectbox('Sex', ['Male', 'Female'])
fasting_glucose = st.number_input('Fasting Blood Glucose', min_value=0)
data = {
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'SkinThickness': skin_thickness,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': diabetes_pedigree,
    'Age': age,
    'Sex': 1 if sex == 'Male' else 0,  # assuming label encoding
    'Fasting_Blood_Glucose': fasting_glucose
}

df = pd.DataFrame(data, index=[0])

df_scaled = scaler.transform(df[columns_to_scale])

if st.button('Predict'):
    result = model.predict(df_scaled)
    if result[0] == 1:
        st.error("Positive for Diabetes")
    else:
        st.success("Negative for Diabetes")