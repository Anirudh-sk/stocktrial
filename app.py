from flask import Flask, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)

# Load ensemble models for 'Color' and 'Safety'
color_model_path = 'colormodels\AdaBoost_0.7116.h5'  # Replace with the actual path
safety_model_path = 'safetymodels\AdaBoost_1.0000.h5'  # Replace with the actual path

color_model = joblib.load(color_model_path)
safety_model = joblib.load(safety_model_path)

# HTML form to accept input features
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        features = [
            float(request.form['open']),
            float(request.form['close']),
            float(request.form['high']),
            float(request.form['low']),
            float(request.form['adjclose']),
            float(request.form['volume']),
            float(request.form['magnitude'])
        ]

        # Reshape the features for prediction
        features = np.array(features).reshape(1, -1)

        # Predict 'Color'
        color_prediction = color_model.predict(features)[0]

        # Predict 'Safety'
        safety_prediction = safety_model.predict(features)[0]

        prediction = {
            'Color': 'Green' if color_prediction == 0 else 'Red',
            'Safety': 'Safe' if safety_prediction == 1 else 'Unsafe'
        }

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
