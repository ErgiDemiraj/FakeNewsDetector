from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Paths to your saved model + vectorizer
MODEL_PATH = os.path.join("model", "fakenews_model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

# Load the trained fake news model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_text = request.form.get("user_text", "")

    if not user_text.strip():
        return render_template(
            "index.html",
            user_text=user_text,
            prediction="Please enter some text to analyze."
        )

    # Transform text and make prediction
    X_vec = vectorizer.transform([user_text])
    pred = model.predict(X_vec)[0]
    prob = model.predict_proba(X_vec)[0].max()

    # Assuming: 1 = fake, 0 = real
    if pred == 1:
        label = f"Prediction: FAKE NEWS (confidence: {prob:.2f})"
    else:
        label = f"Prediction: REAL / NOT FAKE (confidence: {prob:.2f})"

    return render_template("index.html", user_text=user_text, prediction=label)

if __name__ == "__main__":
    app.run(debug=True)
