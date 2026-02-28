import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    roc_curve
)

print("Loading datasets...")

fake = pd.read_csv("dataset/Fake.csv")
fake["label"] = 0

real = pd.read_csv("dataset/True.csv")
real["label"] = 1

df = pd.concat([fake, real], ignore_index=True)

df["content"] = df["title"] + " " + df["text"]

X = df["content"]
y = df["label"]

vectorizer = TfidfVectorizer(stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("\n===== UNIQUE METRICS =====")
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Log Loss:", log_loss(y_test, y_prob))
print("MCC:", matthews_corrcoef(y_test, y_pred))
print("Cohen Kappa:", cohen_kappa_score(y_test, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.show()

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained successfully!")