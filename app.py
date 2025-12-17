from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load models
scaler = joblib.load("scaler.pkl")
model = joblib.load("svm_ova_model.pkl")

SEGMENT_INFO = {
    0: {
        "name": "Low-Value Customer",
        "action": "Discount campaigns and mass promotions"
    },
    1: {
        "name": "Medium-Value Customer",
        "action": "Email marketing and bundled offers"
    },
    2: {
        "name": "High-Value Customer",
        "action": "VIP rewards and personalized offers"
    }
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        X = np.array([[
            float(data["annual_income"]),
            float(data["spend_wine"]),
            float(data["spend_meat"])
        ]])

        X_scaled = scaler.transform(X)
        prediction = int(model.predict(X_scaled)[0])

        return jsonify({
            "success": True,
            "segment": prediction,
            "segment_name": SEGMENT_INFO[prediction]["name"],
            "action": SEGMENT_INFO[prediction]["action"]
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
