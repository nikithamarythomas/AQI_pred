from flask import Flask, jsonify

app = Flask(__name__)

# Assume you have a function called predict_aqi() that returns the AQI prediction
def predict_aqi():
    # Your prediction logic goes here
    # For demonstration purposes, let's assume a constant prediction
    predicted_aqi = 50
    return predicted_aqi

@app.route('/')
def get_aqi():
    predicted_aqi = predict_aqi()
    return jsonify({'AQI Prediction': predicted_aqi})

if __name__ == '__main__':
    app.run(debug=True)