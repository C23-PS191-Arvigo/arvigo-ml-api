# Base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the Flask app files to the container's working directory
COPY . /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install the required packages
RUN pip install flask joblib numpy tensorflow opencv-python mediapipe requests scikit-learn pandas keras

# Expose the port on which the Flask app will run
EXPOSE 80

# # Set the environment variables
# ENV FLASK_APP=server.py
# ENV FLASK_RUN_HOST=0.0.0.0

# Command to run the Flask app
CMD ["flask", "run"]
