import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("student_model.pkl")

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 Student Performance Predictor")
st.write("Enter the student's details below to predict the final grade.")

st.markdown("---")

# Input fields
study_hours = st.slider(
    "Study Hours Per Week",
    min_value=0,
    max_value=60,
    value=20
)

attendance = st.slider(
    "Attendance Rate (%)",
    min_value=0,
    max_value=100,
    value=80
)

previous_grade = st.slider(
    "Previous Grade",
    min_value=0,
    max_value=100,
    value=70
)

# Prediction button
if st.button("Predict Performance"):

    input_data = np.array([[study_hours, attendance, previous_grade]])

    prediction = model.predict(input_data)[0]

    st.success(f"🎯 Predicted Final Grade: {prediction:.2f}")