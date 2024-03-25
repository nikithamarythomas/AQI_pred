from flask import Flask, jsonify
from tensorflow.keras.models import load_model
import numpy as np

# Load the Keras model
model = load_model("my_model.keras")

# Create a Flask app
app = Flask(__name__)

# Sample input data (replace this with your actual input data)
sample_input_data = np.random.rand(1, 10, 9)  # Example: 1 sequence of length 10 with 9 features

@app.route('/')
def display_output():
    # Perform prediction with sample input data
    output = model.predict(sample_input_data)
    
    # Display the output
    return f"Predicted Output: {output}"

if __name__ == '__main__':
    app.run(debug=True)