# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# This includes the src/, templates/, etc. directories based on .dockerignore
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable (optional, can be useful)
# ENV NAME World

# Ensure the python path includes the src directory for imports
ENV PYTHONPATH=/app

# Create the logging directory so the app doesn't fail on first run if volume isn't mounted
RUN mkdir -p /app/logging

# Run web_app.py when the container launches
# Use python -m to ensure modules are found correctly
# Use --host=0.0.0.0 to make it accessible from outside the container
CMD ["python", "-m", "src.calculator_app.web_app", "--host=0.0.0.0"] 