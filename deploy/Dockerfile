# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the ports for FastAPI and mlflow
EXPOSE 8000 5000

# Default command to run the FastAPI app and mlflow tracking server
CMD ["sh", "-c", "uvicorn Predict:app --host 0.0.0.0 --port 8000 & mlflow server --host 0.0.0.0 --port 5000"]
