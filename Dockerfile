# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask application code
COPY . .

# Install the local project as a dependency
RUN pip install --no-cache-dir -e .

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Expose the port Flask will run on
EXPOSE 5000

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]