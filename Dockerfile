# Use the official Python base image with Alpine Linux
FROM python:3.8-alpine

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the required packages
RUN apk add --no-cache gcc musl-dev linux-headers

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Expose the port on which the Flask app will run
EXPOSE 80

# Run the Flask application
CMD ["python", "server.py"]
