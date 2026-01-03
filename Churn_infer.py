# churn_infer.py
import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("churn_model.joblib")

# Example customer record
sample = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 1020.0
}])

# Predict churn
proba = model.predict_proba(sample)[:, 1][0]
pred = model.predict(sample)[0]

print(f"Churn probability: {proba:.3f}")
print("Predicted class:", "Yes" if pred == 1 else "No")