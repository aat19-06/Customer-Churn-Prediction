# churn_train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
import joblib

#  Load dataset
df =pd.read_csv("telco_churn.csv")

# Clean up TotalCharges if needed
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

#  Target + features
y = df["Churn"].map({"Yes": 1, "No": 0}).values
X = df.drop(columns=["Churn"])

#  Identify column types
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

#  Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocess = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

#  Model
clf = LogisticRegression(max_iter=2000, class_weight="balanced")

#  Pipeline
model = Pipeline([
    ("prep", preprocess),
    ("clf", clf)
])

#  Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#  Fit model
model.fit(X_train, y_train)

#  Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

#  Save model
joblib.dump(model, "churn_model.joblib")
print("Model saved as churn_model.joblib")