from flask import Flask, request, jsonify
import joblib
import numpy as np
import random
import tensorflow as tf
import base64
from keras.models import load_model
import pandas as pd
import cv2
import mediapipe as mp

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

def preprocess_image(param):
    # Get the base64 image data from the request
    data = request.json['image']

    # Decode base64 string into image data
    image_data = base64.b64decode(param)

    # Convert image data to NumPy array
    image_np = np.frombuffer(image_data, np.uint8)

    # Read the image using OpenCV
    original = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Resize the image to 300x300
    original = cv2.resize(original, (300, 300))

    # Convert image to RGB
    image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Load the best model from the checkpoint
    best_model = load_model("./models/Arvigo_Landmark.h5")

    mp_face_mesh = mp.solutions.face_mesh

    # Initialize FaceMesh detector
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:

        # Process image
        results = face_mesh.process(image)

        # Get face landmarks
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
        else:
            landmarks = None

        # Create black image with the same size as the original image
        image_height, image_width, _ = image.shape
        black_image = np.zeros((image_height, image_width, 3), np.uint8)

        # Draw landmarks on the black image and get outermost points
        landmark_points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            landmark_points.append((x, y))
        hull = cv2.convexHull(np.array(landmark_points))
        cv2.drawContours(black_image, [hull], -1, (0, 0, 255), 2)

        # Reshape the input for the model prediction
        test_input = np.array(landmark_points).reshape(1, -1)

        # Predict the landmarks on the test image using the best model
        predicted_landmarks = best_model.predict(test_input)

        # Draw the predicted landmarks on the black image
        for landmark in predicted_landmarks.reshape(-1, 2):
            x = int(landmark[0])
            y = int(landmark[1])
            cv2.circle(black_image, (x, y), 2, (0, 255, 0), -1)
        
        # Set alpha channel based on sigmoid function
        for i in range(image_height):
            alpha = 255 / (1 + np.exp(-10 * ((i / image_height) - 0.5)))
            for j in range(image_width):
                if cv2.pointPolygonTest(hull, (j, i), False) >= 0:
                    black_image[i, j] = original[i, j]
                else:
                    black_image[i, j] = np.clip(original[i, j] - alpha, 0, 255)

    # Encode the result image to base64
    _, img_encoded = cv2.imencode('.png', black_image)
    result_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Return the base64-encoded image as the response
    return result_base64

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
        processed_image = preprocess_image(param)        
        
        model = load_model("./models/face-shapes.h5")
        img = load_image_from_base64(processed_image)
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
    if request.content_type == "application/json":
        try:
            data = request.get_json()

            input_data = np.array([list(data.values())], dtype=np.float64)
            df = pd.DataFrame(input_data, columns=data.keys())
            cols_to_modify = ['EXT2','EXT4','EXT6','EXT8','EXT10',
                            'EST2','EST4',
                            'AGR1','AGR3','AGR5','AGR7',
                            'CSN2','CSN4','CSN6','CSN8',
                            'OPN2','OPN4','OPN6','OPN9']
            
            df[cols_to_modify] = df[cols_to_modify].apply(lambda x: 6 - x)

            df['EXT'] = df.filter(regex='EXT\d+').sum(axis=1)
            df['EST'] = df.filter(regex='EST\d+').sum(axis=1)
            df['AGR'] = df.filter(regex='AGR\d+').sum(axis=1)
            df['CSN'] = df.filter(regex='CSN\d+').sum(axis=1)
            df['OPN'] = df.filter(regex='OPN\d+').sum(axis=1)
            df = df.iloc[:, 50:]
            df = df.apply(lambda x: (x - 10) / 40)

            sums = df.iloc[0]
            percentages = [round((x/sums.sum())*100, 2) for x in sums]
            
            per_ext = percentages[0]
            per_est = percentages[1]
            per_agr = percentages[2]
            per_csn = percentages[3]
            per_opn = percentages[4]

            response = {
                'percentage_of_extraversion': per_ext,
                'percentage_of_neurotic': per_est,
                'percentage_of_agreeable': per_agr,
                'percentage_of_conscientious': per_csn,
                'percentage_of_openess': per_opn
            }

            return jsonify(response)

        except:
            return jsonify({'error': 'Failed to process the request.'})

    else:
        return jsonify({'error': 'Invalid content type. Expected application/json.'})


if __name__ == "__main__":
    app.run(debug=True)