# Use Python 3.10 base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into the container
COPY . .

# Default command to run when container starts
CMD ["python", "prediction.py"]
