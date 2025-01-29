# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the app code into the container
COPY main.py .
COPY requirements.txt .
COPY templates/ /app/templates/

# Install required Python packages
RUN pip install -r requirements.txt

# Expose port 8000 for the UI server
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
