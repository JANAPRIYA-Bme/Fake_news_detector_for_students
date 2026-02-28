from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    data = vectorizer.transform([news])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][prediction]

    if prediction == 1:
        result = "✅ Real News"
    else:
        result = "❌ Fake News"

    confidence = round(probability*100,2)

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)