import joblib
import pandas as pd

# Load the trained model
model = joblib.load("student_model.pkl")

print("===== Student Performance Predictor =====")

# Take input from user
study_hours = float(input("Enter Study Hours Per Week: "))
attendance = float(input("Enter Attendance Rate: "))
previous_grade = float(input("Enter Previous Grade: "))

# Create a DataFrame
new_student = pd.DataFrame({
    "StudyHoursPerWeek": [study_hours],
    "AttendanceRate": [attendance],
    "PreviousGrade": [previous_grade]
})

# Predict Final Grade
prediction = model.predict(new_student)

print("\nPredicted Final Grade:", round(prediction[0], 2))