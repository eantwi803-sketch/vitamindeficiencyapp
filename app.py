
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("vitamin_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Vitamin Deficiency Detection System")
st.write("Enter your details below:")

# Basic Info
age = st.number_input("Age", min_value=18, max_value=84)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", min_value=15.0, max_value=45.0, value=25.0)

# Lifestyle
smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
alcohol_consumption = st.selectbox("Alcohol Consumption", ["None", "Moderate", "Heavy"])
exercise_level = st.selectbox("Exercise Level", ["Sedentary", "Light", "Moderate", "Active"])
diet_type = st.selectbox("Diet Type", ["Vegan", "Vegetarian", "Pescatarian", "Omnivore"])
sun_exposure = st.selectbox("Sun Exposure", ["Low", "Moderate", "High"])
income_level = st.selectbox("Income Level", ["Low", "Medium", "High"])
latitude_region = st.selectbox("Latitude Region", ["Tropical", "Subtropical", "Temperate"])

# Vitamin & Blood Levels
vitamin_a = st.number_input("Vitamin A % RDA", 0.0, 200.0, 80.0)
vitamin_c = st.number_input("Vitamin C % RDA", 0.0, 200.0, 75.0)
vitamin_d = st.number_input("Vitamin D % RDA", 0.0, 200.0, 60.0)
vitamin_e = st.number_input("Vitamin E % RDA", 0.0, 200.0, 90.0)
vitamin_b12 = st.number_input("Vitamin B12 % RDA", 0.0, 200.0, 85.0)
folate = st.number_input("Folate % RDA", 0.0, 200.0, 70.0)
calcium = st.number_input("Calcium % RDA", 0.0, 200.0, 95.0)
iron = st.number_input("Iron % RDA", 0.0, 200.0, 88.0)
hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 14.5)
serum_vitamin_d = st.number_input("Serum Vitamin D (ng/mL)", 0.0, 100.0, 25.0)
serum_vitamin_b12 = st.number_input("Serum Vitamin B12 (pg/mL)", 0.0, 1000.0, 300.0)
serum_folate = st.number_input("Serum Folate (ng/mL)", 0.0, 30.0, 10.0)

# Symptoms
symptoms_count = st.number_input("Number of Symptoms", min_value=0, max_value=10, step=1)
symptoms_list = st.number_input("Symptoms List (encoded)", min_value=0, max_value=20, step=1)
has_night_blindness = st.selectbox("Night Blindness", ["No", "Yes"])
has_fatigue = st.selectbox("Fatigue", ["No", "Yes"])
has_bleeding_gums = st.selectbox("Bleeding Gums", ["No", "Yes"])
has_bone_pain = st.selectbox("Bone Pain", ["No", "Yes"])
has_muscle_weakness = st.selectbox("Muscle Weakness", ["No", "Yes"])
has_numbness_tingling = st.selectbox("Numbness/Tingling", ["No", "Yes"])
has_memory_problems = st.selectbox("Memory Problems", ["No", "Yes"])
has_pale_skin = st.selectbox("Pale Skin", ["No", "Yes"])
disease_diagnosis = st.number_input("Disease Diagnosis (encoded)", min_value=0, max_value=4, step=1)

# Encode inputs
gender_enc = 1 if gender == "Male" else 0
smoking_enc = ["Never", "Former", "Current"].index(smoking_status)
alcohol_enc = ["None", "Moderate", "Heavy"].index(alcohol_consumption)
exercise_enc = ["Sedentary", "Light", "Moderate", "Active"].index(exercise_level)
diet_enc = ["Vegan", "Vegetarian", "Pescatarian", "Omnivore"].index(diet_type)
sun_enc = ["Low", "Moderate", "High"].index(sun_exposure)
income_enc = ["Low", "Medium", "High"].index(income_level)
lat_enc = ["Tropical", "Subtropical", "Temperate"].index(latitude_region)

def yn(val): return 1 if val == "Yes" else 0

if st.button("Predict"):
    input_data = pd.DataFrame([{
        'age': age, 'gender': gender_enc, 'bmi': bmi,
        'smoking_status': smoking_enc, 'alcohol_consumption': alcohol_enc,
        'exercise_level': exercise_enc, 'diet_type': diet_enc,
        'sun_exposure': sun_enc, 'income_level': income_enc,
        'latitude_region': lat_enc,
        'vitamin_a_percent_rda': vitamin_a, 'vitamin_c_percent_rda': vitamin_c,
        'vitamin_d_percent_rda': vitamin_d, 'vitamin_e_percent_rda': vitamin_e,
        'vitamin_b12_percent_rda': vitamin_b12, 'folate_percent_rda': folate,
        'calcium_percent_rda': calcium, 'iron_percent_rda': iron,
        'hemoglobin_g_dl': hemoglobin, 'serum_vitamin_d_ng_ml': serum_vitamin_d,
        'serum_vitamin_b12_pg_ml': serum_vitamin_b12, 'serum_folate_ng_ml': serum_folate,
        'symptoms_count': symptoms_count, 'symptoms_list': symptoms_list,
        'has_night_blindness': yn(has_night_blindness), 'has_fatigue': yn(has_fatigue),
        'has_bleeding_gums': yn(has_bleeding_gums), 'has_bone_pain': yn(has_bone_pain),
        'has_muscle_weakness': yn(has_muscle_weakness),
        'has_numbness_tingling': yn(has_numbness_tingling),
        'has_memory_problems': yn(has_memory_problems),
        'has_pale_skin': yn(has_pale_skin), 'disease_diagnosis': disease_diagnosis
    }])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    result = "Has Multiple Deficiencies" if prediction[0] == 1 else "No Multiple Deficiencies"
    st.success(f"Prediction: **{result}**")