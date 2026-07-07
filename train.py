import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("student_performance_updated_1000.csv")

# Remove missing values
data = data.dropna()

# Select input features
X = data[["StudyHoursPerWeek", "AttendanceRate", "PreviousGrade"]]

# Select output
y = data["FinalGrade"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Predict FinalGrade
y_pred = model.predict(X_test)

print("\nFirst 5 Predictions:")
print(y_pred[:5])

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Check model performance
print("\nMean Absolute Error:", mae)
print("R² Score:", r2)

# Plot Actual vs Predicted
plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Final Grade")
plt.ylabel("Predicted Final Grade")
plt.title("Actual vs Predicted Final Grades")

plt.show()
# Calculate errors
errors = y_test - y_pred

plt.figure(figsize=(8,6))

plt.hist(errors, bins=20)

plt.xlabel("Prediction Error")
plt.ylabel("Number of Students")
plt.title("Distribution of Prediction Errors")

plt.show()
plt.figure(figsize=(8,6))

plt.scatter(data["StudyHoursPerWeek"], data["FinalGrade"])

plt.xlabel("Study Hours Per Week")
plt.ylabel("Final Grade")
plt.title("Study Hours vs Final Grade")

plt.show()


# save the trained model
joblib.dump(model, "student_model.pkl")
print("\nModel saved successfully as student_model.pkl")
plt.savefig("images/actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.show()