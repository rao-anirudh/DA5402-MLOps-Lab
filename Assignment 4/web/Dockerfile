# Use lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and templates
COPY app.py .
COPY templates/ templates/

# Expose the default Flask port
EXPOSE 80

# Run the Flask app
CMD ["python", "app.py"]
