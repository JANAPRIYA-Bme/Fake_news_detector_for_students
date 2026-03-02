import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

# ---------- LOAD DATA ----------

fake = pd.read_csv("dataset/fake.csv")
real = pd.read_csv("dataset/true.csv")

fake["label"] = 0
real["label"] = 1

df = pd.concat([fake, real])
df = df.sample(frac=1)

X = df["text"]
y = df["label"]

# ---------- TF-IDF ----------

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vect = vectorizer.fit_transform(X)

# ---------- SPLIT ----------

X_train, X_test, y_train, y_test = train_test_split(
    X_vect, y, test_size=0.2, random_state=42
)

# ---------- MODEL ----------

model = LogisticRegression()
model.fit(X_train, y_train)

# ---------- PREDICTION ----------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ---------- METRICS ----------

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

cv = cross_val_score(model, X_vect, y, cv=5)
cv_mean = cv.mean()

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("ROC AUC:", roc)
print("Cross Validation:", cv_mean)

# ---------- SAVE MODEL ----------

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# ---------- ROC CURVE ----------

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

trust_score = (accuracy + f1 + roc + cv_mean) / 4
print("Final Trust Score:", trust_score)

print("\n==============================")
print("Model Training Completed Successfully ✅")
print("==============================")
print(f"Accuracy        : {accuracy:.4f}")
print(f"F1 Score        : {f1:.4f}")
print(f"ROC AUC         : {roc:.4f}")
print(f"Cross Validation: {cv_mean:.4f}")
print("------------------------------")
print(f"Final Trust Score: {trust_score}%")
print("==============================\n")