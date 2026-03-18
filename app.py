from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'model': request.form['model'],
        'vehicle_age': int(request.form['vehicle_age']),
        'km_driven': int(request.form['km_driven']),
        'seller_type': request.form['seller_type'],
        'fuel_type': request.form['fuel_type'],
        'transmission_type': request.form['transmission_type'],
        'mileage': float(request.form['mileage']),
        'engine': int(request.form['engine']),
        'max_power': int(request.form['max_power']),
        'seats': int(request.form['seats'])
    }

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return render_template("index.html", prediction_text=f"Estimated Price: ₹ {round(prediction, 2)}")

import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))