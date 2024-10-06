import streamlit as st
import pickle
import numpy as np
from pycaret.classification import * 

# Load models
model_anxiety = load_model('model_anxiety')
model_stress = load_model('model_stress')
model_depression = load_model('model_depression')

# Streamlit App
st.title("🌿Mental Health Prediction App🌿")
st.write("""
Welcome to the Mental Health Prediction App! This tool is designed to assess your likelihood of experiencing anxiety, stress, and depression. 
By providing some details about your personal and academic life, you can receive insights tailored to your current mental health status. 
Please fill out the form below to get started.
""")


# User input
st.header("📋 Personal Information")
# Daftar pilihan yang ditampilkan di selectbox
display_options_age = ["18-22", "23-26", "27-30"]
age = st.selectbox("Select your age range:", options=display_options_age)
value_options_age = [1, 2, 3]
age = value_options_age[display_options_age.index(age)]

display_options_gender = ["Male", "Female"]
gender = st.selectbox("Select your gender:", options=display_options_gender)
value_options_gender = [1, 2]
gender = value_options_gender[display_options_gender.index(gender)]

department = st.selectbox(
    "Select your department:",
    [
        'CS/IT Engineering',
        'Other',
        'EEE/ECE Engineering',
        'Env/Life Sciences',
        'Business/Entrepreneurship',
        'Biological Sciences',
        'Civil Engineering',
        'Mechanical Engineering'
    ]
)

st.header("📚 Academic Information")
display_options_academic = ["First Year", "Second Year", "Third Year", "Fourth Year"]
academic_year = st.selectbox("Select your academic year:", options=display_options_academic)
value_options_academic = [1, 2, 3, 4]
academic_year = value_options_academic[display_options_academic.index(academic_year)]

display_options_cgpa = ["Below 2.50", "2.50 - 2.99", "3.00 - 3.39", "3.40 - 3.79", "3.80 - 4.00", "Other"]
cgpa = st.selectbox("Select your CGPA range:", options=display_options_cgpa)
value_options_cgpa = [1, 2, 3, 4, 5, 6]
cgpa = value_options_cgpa[display_options_cgpa.index(cgpa)]

display_options_scholarship = ["Yes", "No"]
scholarship = st.selectbox("Are you receiving a scholarship?", display_options_scholarship)
value_options_scholarship = [1, 2]
scholarship = value_options_scholarship[display_options_scholarship.index(scholarship)]

# Frequency inputs using sliders
st.header("🧠 Psychological Factors")
st.subheader("Scale: 0 = Never, 1 = Rarely, 2 = Sometimes, 3/4 = Very Often")

nervous_frequency = st.radio("How often do you feel nervous?", [0, 1, 2, 3])
worry_frequency = st.radio("How often do you worry?", [0, 1, 2, 3])
relaxation_frequency = st.radio("How often can you relax?", [0, 1, 2, 3])
irritation_frequency = st.radio("How often do you feel irritated?", [0, 1, 2, 3])
overthinking_frequency = st.radio("How often do you overthink?", [0, 1, 2, 3])
restlessness_frequency = st.radio("How often do you feel restless?", [0, 1, 2, 3])
fear_frequency = st.radio("How often do you feel afraid?", [0, 1, 2, 3])
upset_frequency = st.radio("How often do you get upset?", [0, 1, 2, 3, 4])
control_frequency = st.radio("How often can you control your emotions?", [0, 1, 2, 3, 4])
stress_frequency = st.radio("How often do you feel stressed?", [0, 1, 2, 3, 4])
coping_frequency = st.radio("How well can you cope with stress?", [0, 1, 2, 3, 4])
confidence_frequency = st.radio("How confident do you feel?", [0, 1, 2, 3, 4])
life_control_frequency = st.radio("How well do you feel in control of your life?", [0, 1, 2, 3, 4])
irritation_control_frequency = st.radio("How well can you control your irritation?", [0, 1, 2, 3, 4])
top_performance_frequency = st.radio("How often do you feel at the top of your performance?", [0, 1, 2, 3, 4])
anger_frequency = st.radio("How often do you feel angry?", [0, 1, 2, 3, 4])
overwhelm_frequency = st.radio("How often do you feel overwhelmed?", [0, 1, 2, 3, 4])
no_interest_frequency = st.radio("How often do you lose interest in activities?", [0, 1, 2, 3])
hopeless_frequency = st.radio("How often do you feel hopeless?", [0, 1, 2, 3])
sleep_issues_frequency = st.radio("How often do you have trouble sleeping?", [0, 1, 2, 3])
low_energy_frequency = st.radio("How often do you feel low energy?", [0, 1, 2, 3])
appetite_issues_frequency = st.radio("How often do you have appetite issues?", [0, 1, 2, 3])
self_worth_frequency = st.radio("How often do you feel a lack of self-worth?", [0, 1, 2, 3])
concentration_issues_frequency = st.radio("How often do you have trouble concentrating?", [0, 1, 2, 3])
slow_movement_frequency = st.radio("How often do you feel slow in your movements?", [0, 1, 2, 3])
suicidal_thoughts_frequency = st.radio("How often do you have suicidal thoughts?", [0, 1, 2, 3])

# Create input array for the model
input_data = np.array([[age, gender, department, academic_year, cgpa, scholarship,
                        nervous_frequency, worry_frequency, relaxation_frequency, irritation_frequency,
                        overthinking_frequency, restlessness_frequency, fear_frequency, upset_frequency,
                        control_frequency, stress_frequency, coping_frequency, confidence_frequency,
                        life_control_frequency, irritation_control_frequency, top_performance_frequency,
                        anger_frequency, overwhelm_frequency, no_interest_frequency, hopeless_frequency,
                        sleep_issues_frequency, low_energy_frequency, appetite_issues_frequency,
                        self_worth_frequency, concentration_issues_frequency, slow_movement_frequency,
                        suicidal_thoughts_frequency]])

import pandas as pd 
df_2 = pd.DataFrame(input_data)
df_2.columns = ['Age', 'Gender', 'Department', 'AcademicYear', 'CGPA', 'Scholarship',
 'NervousFreq', 'WorryFreq', 'RelaxationFreq', 'IrritationFreq',
 'OverthinkingFreq', 'RestlessnessFreq', 'FearFreq', 'UpsetFreq',
 'ControlFreq', 'StressFreq', 'CopingFreq', 'ConfidenceFreq',
 'LifeControlFreq', 'IrritationControlFreq', 'TopPerformanceFreq',
 'AngerFreq', 'OverwhelmFreq', 'NoInterestFreq', 'HopelessFreq',
 'SleepIssuesFreq', 'LowEnergyFreq', 'AppetiteIssuesFreq',
 'SelfWorthFreq', 'ConcentrationIssuesFreq', 'SlowMovementFreq',
 'SuicidalThoughtsFreq']

# Convert categorical data to numerical format if necessary
# This part depends on how your model was trained. Update accordingly.

# Make predictions
if st.button("💡 Predict"):
    # Perform predictions
    anxiety_prediction = predict_model(model_anxiety, df_2)
    stress_prediction = predict_model(model_stress, df_2)
    depression_prediction = predict_model(model_depression, df_2)

    # Get prediction labels
    anxiety_label = anxiety_prediction['prediction_label'].iloc[0]  
    stress_label = stress_prediction['prediction_label'].iloc[0]    
    depression_label = depression_prediction['prediction_label'].iloc[0]  

    # Display results
    st.subheader("🔍 Prediction Results")
    st.write(f"Anxiety Level: {anxiety_label}")
    st.write(f"Stress Level: {stress_label}")
    st.write(f"Depression Level: {depression_label}")
