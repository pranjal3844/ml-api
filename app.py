from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "ML API is running"

@app.route("/predict", methods=["POST"])
def predict():
    
    data = request.json["features"]
    
    prediction = model.predict([data])

    return jsonify({
        "prediction": prediction.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)