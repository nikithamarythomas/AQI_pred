from flask import Flask, render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd


app = Flask(__name__)

# Load the dataset
df = pd.read_csv("air_pollution_data.csv")

# Preprocessing
# Assuming 'dt' is the timestamp column
df['dt'] = pd.to_datetime(df['dt'])
df.set_index('dt', inplace=True)

# Define the sequence length
sequence_length = 10

# Create overlapping sequences and corresponding target sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])  # Target is the next element after the sequence
    return np.array(X), np.array(y)

# Create input sequences and corresponding target sequences
X, y = create_sequences(df.values, sequence_length)

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, sequence_length * X.shape[2])).reshape(X.shape)
y_scaled = scaler.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Load the model
model = load_model("my_model.keras")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_aqi')
def predict_aqi():
    # Prepare the input data for prediction
    num_predictions = 1
    input_data = X_test[-1]  # Use the last sequence from the test set as input

    # Reshape the input data to match the model's input shape
    reshaped_input_data = input_data.reshape(1, sequence_length, X.shape[2])

    # Make predictions
    predicted_values = []
    for _ in range(num_predictions):
        # Predict the next AQI value
        next_prediction = (model.predict(reshaped_input_data)[0][0])*10
        print(next_prediction)
        #Round the predicted value to the nearest integer
        rounded_prediction = round(next_prediction)
        print(rounded_prediction)
        #Store the predicted value
        predicted_values.append(rounded_prediction)
        # Update the input data for the next prediction
        input_data = np.append(input_data[1:], next_prediction)

    # Format the predicted AQI values
    predicted_aqi = [value for value in predicted_values]

    # Return the predicted AQI values as JSON
    return jsonify({"predicted_values": (predicted_aqi)})

if __name__ == '__main__':
    app.run(debug=True)
