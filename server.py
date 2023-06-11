from flask import Flask, request, jsonify, Response
import joblib
import numpy as np
import random
import tensorflow as tf
import base64
from keras.models import load_model
import pandas as pd
import cv2
import mediapipe as mp
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams
from nltk.metrics.distance import edit_distance
from scipy.sparse import hstack
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
API_KEY = '05d27624-e862-4127-afc8-d382a137ec52'

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

        # Create black image with same size as original image
        image_height, image_width, _ = image.shape
        black_image = np.zeros((image_height, image_width, 3), np.uint8)

        # Draw landmarks on black image and get outermost points
        landmark_points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            landmark_points.append((x, y))
        hull = cv2.convexHull(np.array(landmark_points))
        cv2.drawContours(black_image, [hull], -1, (0, 0, 255), 2)

        # Find leftmost, rightmost, topmost, and bottommost landmarks
        leftmost_landmark = min(landmark_points, key=lambda p: p[0])
        rightmost_landmark = max(landmark_points, key=lambda p: p[0])
        topmost_landmark = min(landmark_points, key=lambda p: p[1])
        bottommost_landmark = max(landmark_points, key=lambda p: p[1])

        # Calculate the size of the square crop
        square_size = max(rightmost_landmark[0] - leftmost_landmark[0], bottommost_landmark[1] - topmost_landmark[1])

        # Calculate the center coordinates of the square crop
        center_x = int((leftmost_landmark[0] + rightmost_landmark[0]) / 2)
        center_y = int((topmost_landmark[1] + bottommost_landmark[1]) / 2)

        # Calculate the shift and expansion based on image size
        shift_percentage = 0.075  # Adjust the shift percentage as desired
        expand_percentage = 0.075  # Adjust the expansion percentage as desired

        shift_pixels = int(shift_percentage * image_height)
        expand_pixels = int(expand_percentage * image_height)

        # Shift the center coordinates upwards
        center_y -= shift_pixels

        # Calculate the coordinates for cropping the square image
        left_crop = center_x - int(square_size / 2)
        right_crop = center_x + int(square_size / 2)
        top_crop = center_y - int(square_size / 2)
        bottom_crop = center_y + int(square_size / 2)

        # Adjust the crop coordinates to ensure they are within the image boundaries
        left_crop = max(0, left_crop - expand_pixels)
        right_crop = min(image_width, right_crop + expand_pixels)
        top_crop = max(0, top_crop - expand_pixels)
        bottom_crop = min(image_height, bottom_crop + expand_pixels)

        # Perform cropping to get the square image
        cropped_image = black_image[top_crop:bottom_crop, left_crop:right_crop]

        # Set alpha channel based on sigmoid function
        for i in range(cropped_image.shape[0]):
            alpha = 255 / (1 + np.exp(-10 * ((i / cropped_image.shape[0]) - 0.5)))
            for j in range(cropped_image.shape[1]):
                if cv2.pointPolygonTest(hull, (j + left_crop, i + top_crop), False) >= 0:
                    cropped_image[i, j] = original[i + top_crop, j + left_crop]
                else:
                    cropped_image[i, j] = np.clip(original[i + top_crop, j + left_crop] - alpha, 0, 255)

        # Resize the square image to 300x300
        cropped_image = cv2.resize(cropped_image, (300, 300))

    # Encode the result image to base64
    _, img_encoded = cv2.imencode('.png', cropped_image)
    result_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Return the base64-encoded image as response
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
    return Response("{'status': 'Healthy'}", status=200, mimetype='application/json')

@app.route('/health_check/ping')
def health_check():
    return Response("{'status': 'Healthy'}", status=200, mimetype='application/json')

@app.route('/is_human', methods=['POST'])
def process_is_human():
    api_key = request.headers.get('X-API-Key')
    if api_key == API_KEY:
        content_type = request.headers.get('Content-Type')
        if content_type == 'application/json':
            param = request.json["image"]
            model = load_model("./models/human-faces.h5")
            img = load_image_from_base64(param)
            pred = predict_image(model, img)
            result = threshold_human(pred)

            if result == "True":
                return jsonify({'result': True})
            elif result == "False":
                return jsonify({'result': False})

        else:
            return jsonify({'message': 'Content-Type not supported!'})
    else:
        return jsonify({'message': 'Invalid API key!'})

@app.route('/face_shape', methods=['POST'])
def process_face_shape():
    api_key = request.headers.get('X-API-Key')
    if api_key == API_KEY:
        content_type = request.headers.get('Content-Type')
        if content_type == 'application/json':
            param = request.json["image"]
            processed_image = preprocess_image(param)   
            
            model = load_model("./models/face-shapes.h5")
            img = load_image_from_base64(processed_image)
            pred = predict_image(model, img)
            result = classify_face_shape(pred)

            # Convert the result keys to strings

            result = {str(key): value for key, value in result.items()}

            return jsonify(result)
        else:
            return 'Content-Type not supported!'
    else:
        return jsonify({'message': 'Invalid API key!'})

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
    api_key = request.headers.get('X-API-Key')
    if api_key == API_KEY:
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
    else:
        return jsonify({'message': 'Invalid API key!'})

@app.route('/product_search', methods=['GET'])
def product_search():
    url = 'https://api.arvigo.site/v1/product-recommendation'
    headers = {'X-API-Key': '4a150010-bac7-46e7-8b8b-594f47b0015c'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        # Mendefinisikan query
        query = request.args.get("query")

        if query is None:
            return jsonify({'message': 'Query parameter is missing'})

        df = pd.DataFrame(data['data'])
        # Menghapus kolom 'clicked'
        df = df.drop('clicked', axis=1)
        # Mengubah semua nilai pada DataFrame menjadi lower case
        df = df.apply(lambda x: x.astype(str).str.lower())
        df['category'] = df['category'].replace({'glasses': 'glasses kacamata'}, regex=True)
        # Mengganti nilai dalam kolom 'tags'
        df['tags'] = df['tags'].replace({'circle': 'circle bulat lingkaran',
                                        'heart': 'heart hati',
                                        'square': 'square kotak',
                                        'triangle': 'triangle segitiga',
                                        'extraversion': 'extraversion ekstrover enerjik terbuka sosial',
                                        'neurotic': 'neurotic emosional labil cemas gugup',
                                        'agreeable': 'agreeable ramah sopan bersahabat bergaul',
                                        'conscientious': 'conscientious bertanggung jawab disiplin cermat teliti',
                                        'openness': 'openness terbuka fleksibel kreatif inovatif'}, regex=True)
        # Pra-pemrosesan teks
        stop_words = set(stopwords.words('indonesian'))
        stemmer = PorterStemmer()
        def preprocess_text(text):
            # Menghapus karakter khusus
            text = re.sub(r'\W', ' ', text)
            
            # Tokenisasi
            tokens = word_tokenize(text)
            
            # Menggabungkan kembali token menjadi kalimat
            processed_text = ' '.join(tokens)
            return processed_text
        df['name'] = df['name'].apply(preprocess_text)
        df['description'] = df['description'].apply(preprocess_text)

        def preprocess_query(query):
            # Mengubah huruf menjadi huruf kecil
            query = query.lower()

            # Menghapus tanda baca
            query = re.sub(r'[^a-zA-Z0-9\s]', '', query)
            return query

        def check_fuzzy_ratio(query_list, korpus):
            result = []
            for word in query_list:
                match = False
                for k in korpus:
                    if fuzz.ratio(word, k) >= 80:
                        result.append(k)
                        match = True
                        break
                if not match:
                    result.append('')
            return result

        def separate_query(query, result_category, korpus):
            query_list = query.split()
            separated_queries = []
            separated_query = ''
            for i, word in enumerate(query_list):
                if result_category[i-1] != '':
                    if separated_query != '':
                        separated_queries.append(separated_query.strip())
                        separated_query = ''
                if result_category[i] != '':
                    separated_query += result_category[i]
                else:
                    corrected_word = None
                    for k in korpus:
                        if fuzz.ratio(word, k) >= 80:
                            corrected_word = k
                            break
                    if corrected_word:
                        separated_query += corrected_word
                    else:
                        separated_query += word
                separated_query += ' '
            if separated_query != '':
                separated_queries.append(separated_query.strip())
            return separated_queries

        def fix_typo_with_fuzzy_ratio(query, korpus):
            corrected_query = query
            words = query.split()
            for i, word in enumerate(words):
                for keyword in korpus:
                    ratio = fuzz.ratio(word, keyword)
                    if ratio > 80:
                        words[i] = keyword
                        break
            corrected_query = ' '.join(words)
            return corrected_query

        def identify_keywords(text_list, korpus_category):
            keywords = []
            for text in text_list:
                keyword = ''
                words = text.split()
                for word in words:
                    if word in korpus_category:
                        keyword = word
                        break
                keywords.append(keyword)
            return keywords

        def neg_merge_lists(list1, list2):
            merged_list = []
            for i in range(len(list1)):
                if list1[i] == '' or list2[i] == '':
                    None
                else:
                    merged_list.append(list2[i])
            return merged_list

        def pos_merge_lists(list1, list2, list3):
            pos_list1 = []
            pos_list2 = []
            for i in range(len(list1)):
                if list1[i] != '' and list3[i] == '':
                    pos_list1.append(list1[i])
                if list2[i] != '' and list3[i] == '':
                    pos_list2.append(list2[i])
            return pos_list1, pos_list2

        korpus_negatif = ['tidak', 'ga', 'gak', 'ngga', 'nggak', 'selain', 'buruk', 'bukan', 'kurang', 'berbeda', 'beda', 'no', 'except', 'not']
        korpus_category = ['jas', 'coat',
                        'kaos', 'shirt', 'tshirt',
                        'blouse', 'blus',
                        'glasses', 'kacamata',
                        'celana', 'pants', 'trousers',
                        'gelang', 'bracelet',
                        'dompet', 'purse',
                        'baju', 'dress',
                        'blazer',
                        'perhiasan', 'jewellery',
                        'tas', 'bag',
                        'rok', 'skirt',
                        'kemeja',
                        'sepatu', 'shoes',
                        'kalung', 'necklace',
                        'sweater',
                        'parfum', 'parfume',
                        'lipstik',
                        'pashmina', 'makeup',
                        'anting', 'earrings',
                        'jam', 'clock', 'watch',
                        'topi', 'hat',
                        'sandal',
                        'jaket', 'jacket',
                        'kemeja']
        korpus_tag = ["circle", "bulat", "lingkaran",
                    "heart","hati",
                    "oblong",
                    "oval", 
                    "square", "kotak", "segiempat",
                    "triangle", "segitiga",
                    "extraversion", "ekstrover", "enerjik", "terbuka", "sosial",
                    "neurotic", "emosional", "labil", "cemas", "gugup",
                    "agreeable", "ramah", "sopan", "bersahabat", "bergaul",
                    "conscientious", "bertanggung jawab", "disiplin", "cermat", "teliti",
                    "openness", "terbuka", "fleksibel", "kreatif", "inovatif"]

        preprocess_queries = preprocess_query(query)
        result_category = check_fuzzy_ratio(preprocess_queries.split(), korpus_category + korpus_tag)
        separated_queries = separate_query(preprocess_queries, result_category, korpus_category + korpus_tag)
        corrected_queries = [fix_typo_with_fuzzy_ratio(query, korpus_negatif) for query in separated_queries]
        keywords_cat = identify_keywords(corrected_queries, korpus_category)
        keywords_tag = identify_keywords(corrected_queries, korpus_tag)
        keywords_neg = identify_keywords(corrected_queries, korpus_negatif)
        neg_category_query = neg_merge_lists(keywords_neg, keywords_cat)
        neg_tag_query = neg_merge_lists(keywords_neg, keywords_tag)
        pos_category_query, pos_tag_query = pos_merge_lists(keywords_cat, keywords_tag, keywords_neg)

        # Membuat fungsi untuk memeriksa apakah tag mengandung kata dalam neg_category_query
        def has_negative_tags(tag, neg_query):
            for query in neg_query:
                if query in tag:
                    return True
            return False

        # Menghapus baris pada DataFrame yang memiliki category yang mengandung kata dalam neg_category_query
        df = df[~df['category'].apply(lambda x: has_negative_tags(x, neg_category_query))]

        # Menghapus baris pada DataFrame yang memiliki tag yang mengandung kata dalam neg_tag_query
        df = df[~df['tags'].apply(lambda x: has_negative_tags(x, neg_tag_query))]

        if len(pos_category_query) != 0:
            df = df[df['category'].apply(lambda x: any(tag in x for tag in pos_category_query))]

        if len(pos_tag_query) != 0:
            df = df[df['tags'].apply(lambda x: any(tag in x for tag in pos_tag_query))]

        # Inisialisasi TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer()

        def search_and_get_results(query, vectorizer, top_n):
            results = []
            for idx, product in df.iterrows():
                # Menghitung similarity score untuk setiap kolom
                similarity_scores = []

                for column in ['name', 'description', 'brand', 'merchants']:
                    query_ngrams = set(ngrams(query.lower(), 3))
                    product_ngrams = set(ngrams(product[column].lower(), 3))
                    ngram_similarity = len(query_ngrams.intersection(product_ngrams)) / len(query_ngrams.union(product_ngrams))

                    levenshtein_distance = edit_distance(query.lower(), product[column].lower())
                    levenshtein_similarity = 1 - (levenshtein_distance / max(len(query), len(product[column])))

                    similarity_score = (ngram_similarity + levenshtein_similarity) / 2

                    # Meningkatkan bobot similarity score pada kolom "category" dan "tags"
                    if column == 'name':
                        similarity_score *= 2
                    elif column == 'description':
                        similarity_score *= 1
                    elif column == 'brand':
                        similarity_score *= 3
                    else:
                        similarity_score *= 2

                    similarity_scores.append(similarity_score)

                # Menjumlahkan similarity scores dari setiap kolom
                total_similarity_score = sum(similarity_scores) / len(similarity_scores)
                results.append((product['id'], total_similarity_score))

            # Mengurutkan hasil berdasarkan similarity score
            results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]

            # Membuat DataFrame hasil pencarian
            search_results = pd.DataFrame(results, columns=['id', 'similarity'])
            return search_results

        # Menampilkan produk berdasarkan query
        search_results = search_and_get_results(query, tfidf_vectorizer, top_n=df.shape[0])
        merged_df = pd.merge(df, search_results, on='id', how='inner')

        # Menampilkan data frame dengan nilai similarity lebih dari 0.1
        merged_df = merged_df[merged_df['similarity'] > 0.1]

        # Mengeprint tidak ada produk apabila merfed_df tidak mempunyai baris
        if merged_df.shape[0] == 0:
            print("Tidak ada produk yang sesuai dengan pencarian\n")

        # Mengurutkan dataframe berdasarkan kolom similarity yang terbesar
        merged_df = merged_df.sort_values(by=['similarity'], ascending=False)

        return jsonify(merged_df.head(25).to_dict(orient='records'))
    else:
        return jsonify({'message': f'Request failed with status code {response.status_code}'})
@app.route('/product_recommendation', methods=['GET'])
def product_recommendation():
    url = 'https://api.arvigo.site/v1/product-recommendation'
    headers = {'X-API-Key': '4a150010-bac7-46e7-8b8b-594f47b0015c'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        api_key = request.headers.get('X-API-Key')
        if api_key == API_KEY:
            data = response.json()

            # Mendefinisikan Banyaknya Rekomendasi Berdasarkan Search
            top_recommendation_item = 25

            # Membuat DataFrame dari data
            data = pd.DataFrame(data['data'])
            data.head()
            # Inisialisasi TF-IDF Vectorizer
            # Menghitung vektor TF-IDF untuk masing-masing komponen dengan bobot yang berbeda
            tfidf_vectorizer = TfidfVectorizer()

            # Menghitung vektor TF-IDF untuk kolom 'name', 'category', 'brand', dan 'description'
            name_tfidf = tfidf_vectorizer.fit_transform(data['name'])
            category_tfidf = tfidf_vectorizer.fit_transform(data['category'])
            brand_tfidf = tfidf_vectorizer.fit_transform(data['brand'])
            description_tfidf = tfidf_vectorizer.fit_transform(data['description'])

            # Menghitung vektor TF-IDF untuk kolom 'tags'
            tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(','), token_pattern=r"(?u)\b\w+\b")
            tags_tfidf = tfidf_vectorizer.fit_transform(data['tags'])
            merchants_tfidf = tfidf_vectorizer.fit_transform(data['merchants'])

            # Menggabungkan matriks TF-IDF dari setiap komponen
            weighted_tfidf_matrix = hstack([
                2 * category_tfidf,
                1.5 * tags_tfidf,
                1.2 * brand_tfidf,
                1 * name_tfidf,
                0.8 * description_tfidf,
                0.5 * merchants_tfidf
            ])

            # Menghitung kemiripan antar produk menggunakan cosine similarity
            similarity_matrix = cosine_similarity(weighted_tfidf_matrix, weighted_tfidf_matrix)

            # Membuat dictionary untuk menyimpan rekomendasi
            recommendations = {}

            # Menampilkan hasil kemiripan

            num_products = len(data)
            for i in range(num_products):
                product_id = str(data.loc[i, 'id'])  # Convert the product_id to string

                similar_products = [(str(data.loc[j, 'id']), similarity_matrix[i, j]) for j in range(num_products) if i != j]
                similar_products = sorted(similar_products, key=lambda x: x[1], reverse=True)[:top_recommendation_item]
                recommendations[product_id] = [product for product, _ in similar_products]

            # Convert the dictionary keys to string and return the JSON response

            recommendations = {str(k): v for k, v in recommendations.items()}
            return jsonify(recommendations)
        else:
            return jsonify({'message': 'Invalid API key!'})
    else:
        return jsonify({'message': f'Request failed with status code {response.status_code}'})

if __name__ == "__main__":
    app.run(debug=True)
