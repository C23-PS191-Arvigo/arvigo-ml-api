from flask import Flask, request, jsonify
import joblib
import numpy as np
import random
import tensorflow as tf
import base64
from keras.models import load_model

app = Flask(__name__)

def load_image_from_base64(base64_string, target_size=(100, 100)):
    img_bytes = base64.b64decode(base64_string)
    img = tf.io.decode_image(img_bytes, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    return img

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def predict_image(model, img):
    pred = model.predict(img)
    return pred[0]

def threshold_human(value):
    non_human_probability = value[0]
    human_probability = value[1]
    threshold = 0.75

    if human_probability >= threshold and non_human_probability <= threshold:
        return str(True)
    else:
        return str(False)

def classify_face_shape(value):
    shapes = ['circle', 'heart', 'oblong', 'oval', 'square', 'triangle']
    probabilities = value.tolist()

    # Get the sorted probabilities array that would sort the probabilities in descending order
    sorted_probabilities_array = np.argsort(probabilities)[::-1]

    # Get the highest and second highest probabilities
    highest_probability = probabilities[sorted_probabilities_array[0]]

    # Get the corresponding shapes using the sorted probabilities array
    highest_shape = shapes[sorted_probabilities_array[0]]

    return {
        'shape': highest_shape,
        'probability': highest_probability,
    }

@app.route("/")
def hello_world():
    return "<p>Blank</p>"

@app.route('/health_check/ping')
def health_check():
    return jsonify({'status': 'Healthy'})

@app.route('/is_human', methods=['POST'])
def process_is_human():
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json':
        param = request.json["image"]

        model = load_model("./models/human-faces.h5")
        img = load_image_from_base64(param)
        pred = predict_image(model, img)
        result = threshold_human(pred)

        return result
    else:
        return 'Content-Type not supported!'

@app.route('/face_shape', methods=['POST'])
def process_face_shape():
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json':
        param = request.json["image"]

        model = load_model("./models/face-shapes.h5")
        img = load_image_from_base64(param)
        pred = predict_image(model, img)
        result = classify_face_shape(pred)

        return result
    else:
        return 'Content-Type not supported!'

@app.route("/dummy_detect_personality", methods=["POST"])
def dummy_detect_personality():
    personality_model = joblib.load("./models/big-five-personality.joblib")
    if request.content_type == "application/json":
        try:
            data = {'EXT1': 0, 'EXT2': 0, 'EXT3': 0, 'EXT4': 0, 'EXT5': 0, 'EXT6': 0, 'EXT7': 0, 'EXT8': 0, 'EXT9': 0,
                    'EXT10': 0, 'EST1': 0, 'EST2': 0, 'EST3': 0, 'EST4': 0, 'EST5': 0, 'EST6': 0, 'EST7': 0, 'EST8': 0,
                    'EST9': 0, 'EST10': 0, 'AGR1': 0, 'AGR2': 0, 'AGR3': 0, 'AGR4': 0, 'AGR5': 0, 'AGR6': 0, 'AGR7': 0,
                    'AGR8': 0, 'AGR9': 0, 'AGR10': 0, 'CSN1': 0, 'CSN2': 0, 'CSN3': 0, 'CSN4': 0, 'CSN5': 0, 'CSN6': 0,
                    'CSN7': 0, 'CSN8': 0, 'CSN9': 0, 'CSN10': 0, 'OPN1': 0, 'OPN2': 0, 'OPN3': 0, 'OPN4': 0, 'OPN5': 0,
                    'OPN6': 0, 'OPN7': 0, 'OPN8': 0, 'OPN9': 0, 'OPN10': 0}

            data = {key: random.randint(1, 5) for key in data}
            # Convert the input data to a numpy array
            input_data = np.array([list(data.values())], dtype=np.float64)

            # Perform clustering prediction
            predictions = personality_model.predict(input_data)

            # Convert the predictions to a list
            personality_traits = predictions.tolist()

            # Map the predicted personality traits to their respective names
            personality_names = ['Extraversion', 'Emotional Stability', 'Agreeableness', 'Conscientiousness', 'Openness']
            predicted_personality = [personality_names[i] for i in personality_traits]

            # Prepare the response JSON
            response = {
                'input': data,
                'predicted_personality': predicted_personality
            }

            return jsonify(response)

        except:
            return jsonify({'error': 'Failed to process the request.'})

    else:
        return jsonify({'error': 'Invalid content type. Expected application/json.'})

@app.route("/detect_personality", methods=["POST"])
def detect_personality():
    personality_model = joblib.load("./models/big-five-personality.joblib")
    if request.content_type == "application/json":
        try:
            data = request.get_json()

            # Convert the input data to a numpy array
            input_data = np.array([list(data.values())], dtype=np.float64)

            # Perform clustering prediction
            predictions = personality_model.predict(input_data)

            # Convert the predictions to a list
            personality_traits = predictions.tolist()

            # Map the predicted personality traits to their respective names
            personality_names = ['Extraversion', 'Emotional Stability', 'Agreeableness', 'Conscientiousness', 'Openness']
            predicted_personality = [personality_names[i] for i in personality_traits]

            response = {
                'predicted_personality': predicted_personality
            }

            return jsonify(response)

        except:
            return jsonify({'error': 'Failed to process the request.'})

    else:
        return jsonify({'error': 'Invalid content type. Expected application/json.'})


if __name__ == "__main__":
    app.run(debug=True)