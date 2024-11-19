from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load pre-trained models (replace with actual paths)
binary_model = tf.keras.models.load_model("models/binary_model.h5")
multi_class_model = tf.keras.models.load_model("models/multi_class_model.h5")
dropout_model = tf.keras.models.load_model("models/dropout_model.h5")
batch_norm_model = tf.keras.models.load_model("models/batch_norm_model.h5")
lstm_model = tf.keras.models.load_model("models/lstm_model.h5")
rnn_model = tf.keras.models.load_model("models/rnn_model.h5")

# Helper function to preprocess images
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224))  # Adjust size according to your model's input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Endpoint for Binary Classification
@app.route('/api/binary-classification', methods=['POST'])
def binary_classification():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    image = preprocess_image(file.read())
    prediction = binary_model.predict(image)
    label = 'Class 0' if prediction[0][0] < 0.5 else 'Class 1'
    return jsonify({'label': label, 'confidence': float(prediction[0][0])})

# Endpoint for Multi-class Classification
@app.route('/api/multi-class-classification', methods=['POST'])
def multi_class_classification():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    image = preprocess_image(file.read())
    prediction = multi_class_model.predict(image)
    label = np.argmax(prediction[0])
    return jsonify({'label': f'Class {label}', 'confidence': float(np.max(prediction[0]))})

# Endpoint for Dropout Model
@app.route('/api/dropout-classification', methods=['POST'])
def dropout_classification():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    image = preprocess_image(file.read())
    prediction = dropout_model.predict(image)
    label = np.argmax(prediction[0])
    return jsonify({'label': f'Class {label}', 'confidence': float(np.max(prediction[0]))})

# Endpoint for Batch Normalization Model
@app.route('/api/batch-norm-classification', methods=['POST'])
def batch_norm_classification():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    image = preprocess_image(file.read())
    prediction = batch_norm_model.predict(image)
    label = np.argmax(prediction[0])
    return jsonify({'label': f'Class {label}', 'confidence': float(np.max(prediction[0]))})

# Endpoint for LSTM Model
@app.route('/api/lstm-classification', methods=['POST'])
def lstm_classification():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    image = preprocess_image(file.read())
    prediction = lstm_model.predict(image)
    label = np.argmax(prediction[0])
    return jsonify({'label': f'Class {label}', 'confidence': float(np.max(prediction[0]))})

# Endpoint for RNN Model
@app.route('/api/rnn-classification', methods=['POST'])
def rnn_classification():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    image = preprocess_image(file.read())
    prediction = rnn_model.predict(image)
    label = np.argmax(prediction[0])
    return jsonify({'label': f'Class {label}', 'confidence': float(np.max(prediction[0]))})

if __name__ == '__main__':
    app.run(debug=True)
