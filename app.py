from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Model metrics
accuracy = 0.96
f1 = 0.95
roc = 0.97
cv = 0.94

trust_score = round((accuracy + f1 + roc + cv) / 4 * 100, 2)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        try:
            news = request.form["news"]

            vect = vectorizer.transform([news])

            pred = model.predict(vect)[0]
            prob = model.predict_proba(vect)[0]

            probability = round(np.max(prob) * 100, 2)

            prediction = "REAL" if pred == 1 else "FAKE"

            print("Prediction:", prediction)
            print("Probability:", probability)

        except Exception as e:
            print("Error:", e)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        trust_score=trust_score
    )

if __name__ == "__main__":
    app.run(debug=True)
