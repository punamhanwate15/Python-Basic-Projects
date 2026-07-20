import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Question 1
filepath="heart.csv"
df=pd.read_csv(filepath)

# print("Independent feature: ")
X = df.drop("HeartDisease", axis=1)
Y=df["HeartDisease"]
X=pd.get_dummies(X,drop_first=True)
# print(X.head())
# print(Y.head())

# # Question 2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=42)

# print("DATA SPLIT")
# print("X_train shape: ",X_train.shape)
# print("X_test shape: ",X_test.shape)
# print("Y_train shape: ",Y_train.shape)
# print("Y_test shape: ",Y_test.shape)

from sklearn.preprocessing import StandardScaler

# Create Scaler
scaler = StandardScaler()

# Fit and Transform Training Data
X_train = scaler.fit_transform(X_train)

# Transform Testing Data
X_test = scaler.transform(X_test)

# # Question 3

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)

# print("\nModel Trained Successfully")
# print(model)

# Question 4
Y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    "Actual": Y_test.reset_index(drop=True),
    "Predicted": Y_pred
})

# print("\nFirst 10 Actual vs Predicted Values")
# print(comparison.head(10))

# # Question 5
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# print("\nConfusion Matrix")
# print(cm)

TN, FP, FN, TP = cm.ravel()

# print("\nTN (True Negative):", TN)
# print("FP (False Positive):", FP)
# print("FN (False Negative):", FN)
# print("TP (True Positive):", TP)


# # Question 6

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# print("*" * 50)
# print("      MODEL EVALUATION REPORT")
# print("*" * 50)
# # Predictions
# y_pred = model.predict(X_test)

# # Accuracy
# accuracy = accuracy_score(Y_test, Y_pred)
# print("Accuracy :", accuracy)

# # Precision
# precision = precision_score(Y_test, Y_pred)
# print("Precision :", precision)

# # Recall
# recall = recall_score(Y_test, Y_pred)
# print("Recall :", recall)

# # F1 Score
# f1 = f1_score(Y_test, Y_pred)
# print("F1 Score :", f1)

# # Classification Report
# print("*" * 50)
# print("Classification Report")
# print("*" * 50)
# print("\nClassification Report:")
# print(classification_report(Y_test, Y_pred))


import joblib


# # Save Model
# joblib.dump(model, "heart_model.pkl")

# # Save Scaler
# joblib.dump(scaler, "scaler.pkl")

# # Save Columns
# joblib.dump(X.columns.tolist(), "columns.pkl")

# print("=" * 50)
# print("      MODEL SAVED SUCCESSFULLY")
# print("=" * 50)
# print("✓ Model File   : heart_model.pkl")
# print("✓ Scaler File  : scaler.pkl")
# print("✓ Columns File : columns.pkl")
# print("=" * 50)
# print("All files have been saved successfully!")
# print("=" * 50)

# # Question 8

import joblib
import pandas as pd

# Load Model
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Sample Patient Data
sample = pd.DataFrame([{
    "age": 45,
    "sex": 1,
    "cp": 2,
    "trestbps": 120,
    "chol": 230,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 2
}])

columns = joblib.load("columns.pkl")

sample = sample.reindex(columns=columns, fill_value=0)



# Scaling
sample = scaler.transform(sample)

# Prediction
prediction = model.predict(sample)

print("="*50)
print("MODEL LOADED SUCCESSFULLY")
print("="*50)

if prediction[0] == 1:
    print("Prediction : Heart Disease : YES")
else:
    print("Prediction : Heart Disease : NO")

print("="*50)