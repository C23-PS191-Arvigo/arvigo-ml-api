# Base image
FROM python:3.9-alpine

# Set the working directory inside the container
WORKDIR /app

# Set environment variable for protobuf
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Copy the Flask app files to the container's working directory
COPY . /app

# Install system dependencies
RUN apk add --no-cache libstdc++ musl-dev libffi-dev openssl-dev gcc g++ make libjpeg-turbo-dev
RUN pip install --no-cache-dir --upgrade pip
# Install the required packages
# RUN pip install --no-cache-dir protobuf flask joblib numpy tensorflow opencv-python mediapipe requests scikit-learn pandas keras nltk nvidia-tensorrt fuzzywuzzy python-dotenv scipy
RUN pip install --no-cache-dir protobuf \
    && pip install --no-cache-dir flask \
    && pip install --no-cache-dir joblib \
    && pip install --no-cache-dir numpy \
    && pip install --no-cache-dir tensorflow \
    && pip install --no-cache-dir opencv-python \
    && pip install --no-cache-dir mediapipe \
    && pip install --no-cache-dir requests \
    && pip install --no-cache-dir scikit-learn \
    && pip install --no-cache-dir pandas \
    && pip install --no-cache-dir keras \
    && pip install --no-cache-dir nltk \
    && pip install --no-cache-dir nvidia-tensorrt \
    && pip install --no-cache-dir fuzzywuzzy \
    && pip install --no-cache-dir python-dotenv \
    && pip install --no-cache-dir scipy

# Expose the port on which the Flask app will run
EXPOSE 80

# Set environment variables
ENV FLASK_APP=server.py
ENV FLASK_RUN_HOST=0.0.0.0

# Command to run the Flask app
CMD ["flask", "run"]
