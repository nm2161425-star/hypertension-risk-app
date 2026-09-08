from flask import Flask, request, jsonify, render_template
import pandas as pd
from xgboost import XGBClassifier

app = Flask(__name__)

# Load XGBoost model
model = XGBClassifier()
model.load_model("xgboost_hypertension_model.json")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    input_data = pd.DataFrame([{
        "BMI": data["BMI"],
        "Family_History": data["Family_History"],
        "Smoking_Status": data["Smoking_Status"],
        "Age": data["Age"],
        "Stress_Score": data["Stress_Score"]
    }])

    probability = float(model.predict_proba(input_data)[0][1])

    if probability < 0.30:
        risk = "Low Risk"
    elif probability < 0.70:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    return jsonify({
        "probability": round(probability, 3),
        "risk": risk
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
