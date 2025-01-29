# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the app code into the container
COPY smolLM2.py .
COPY main.py .
COPY requirements.txt .
COPY cosmo2-tokenizer/ /app/cosmo2-tokenizer/
COPY checkpoint/ /app/checkpoint/
# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8001 for the model server
EXPOSE 8001

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
