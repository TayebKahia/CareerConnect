# Job Recommendation API

This is a FastAPI application that uses a Graph Neural Network (GNN) to recommend jobs based on user skills. The model was trained on IT job data to predict suitable job titles for a given set of skills and technologies.

## Features

- Recommend job titles based on a list of skills and technologies
- Weight different skills based on importance
- Get a list of available technologies and skill categories
- Interactive API documentation via Swagger UI

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- SentenceTransformers 2.2.2+
- FastAPI 0.100.0+
- Uvicorn 0.23.0+
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository or copy the files to your local machine
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the API Server

To run the API server locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at http://localhost:8000

### Using Docker

You can also run the API using Docker:

```bash
# Build the Docker image
docker build -t job-recommendation-api .

# Run the container
docker run -p 8000:8000 job-recommendation-api
```

### API Endpoints

- `GET /`: Basic API information
- `GET /skills/technologies`: List all available technologies
- `GET /skills/categories`: List all available skill categories
- `POST /recommend`: Get job recommendations based on skills

### Example API Request

```python
import requests
import json

# Define some skills
skills = {
    "skills": [
        {"name": "Python", "type": "technology_name", "weight": 1.0},
        {"name": "SQL", "type": "technology_name", "weight": 0.85},
        {"name": "Machine learning", "type": "skill_title", "weight": 0.95},
        {"name": "Data analysis", "type": "skill_title", "weight": 1.0}
    ],
    "top_n": 3
}

# Make the API request
response = requests.post(
    "http://localhost:8000/recommend",
    data=json.dumps(skills),
    headers={'Content-Type': 'application/json'}
)

# Print the recommendations
if response.status_code == 200:
    for i, rec in enumerate(response.json()["recommendations"]):
        print(f"{i+1}. {rec['title']} (Score: {rec['score']:.4f})")
else:
    print(f"Error: {response.json()}")
```

### Testing the API

A test script is provided to quickly check if the API is working:

```bash
python test_api.py
```

## API Documentation

Once the API is running, you can access the interactive Swagger documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Model Information

The job recommendation model uses a Heterogeneous Graph Neural Network (HeteroGNN) with the following components:

- Graph structure with job nodes, technology nodes, and skill category nodes
- Graph Attention Networks (GAT) with multiple attention heads
- SentenceTransformers for embedding skill text
- Weighted skill aggregation for personalized recommendations

## License

This project is licensed under the MIT License - see the LICENSE file for details.
