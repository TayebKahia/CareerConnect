FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for PyTorch Geometric
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and data
COPY app.py .
COPY best_model1.pth .
COPY filtered_IT_data.json .

# Expose port for FastAPI
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
