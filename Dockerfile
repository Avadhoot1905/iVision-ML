# Use a base image with Python
FROM python:3.10-slim

# Set working directory
WORKDIR /

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code and model
COPY app.py .
COPY mobilenetv2.pth .

# Expose port 7860
EXPOSE 7860

# Run Flask app
CMD ["python", "app.py"]