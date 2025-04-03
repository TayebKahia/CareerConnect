"""
Test script to validate the new GNN model recommendation API
"""
import requests
import json
import time


def test_recommend_new_endpoint():
    """Test the /api/recommend-new endpoint"""
    url = "http://localhost:5000/api/recommend-new"

    # Test payload with the new format
    payload = {
        "skills": [
            {"name": "TensorFlow (TF)", "similarity": 1.0,
             "type": "technology"},
            {"name": "PyTorch", "similarity": 1.0, "type": "technology"},
            {"name": "Structured query language (SQL)",
             "similarity": 1.0, "type": "technology"},
            {"name": "Python", "similarity": 1.0, "type": "technology"},
            {"name": "Data visualization software (Data Viz SW)",
             "similarity": 0.8, "type": "technology"},
            {"name": "Web framework software (Web FW)",
             "similarity": 0.7, "type": "technology"},
            {"name": "Amazon Web Services CloudFormation (CloudFormation)",
             "similarity": 0.6, "type": "technology"},
            {"name": "Platform as a service PaaS (PaaS)",
             "similarity": 0.6, "type": "technology"},
            {"name": "Oracle software (Oracle Cloud)",
             "similarity": 0.6, "type": "technology"},
            {"name": "Platform interconnectivity software (Connectivity SW) (Interconnect SW)",
             "similarity": 0.5, "type": "technology"},
            {"name": "IBM InfoSphere DataStage (DataStage)",
             "similarity": 0.5, "type": "technology"},
            {"name": "Informatica Data Explorer (Data Explorer)",
             "similarity": 0.5, "type": "technology"}
        ],
        "top_n": 5
    }

    print(f"Sending request to {url}...")

    # Time the request
    start_time = time.time()

    # Send request
    response = requests.post(url, json=payload)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Print results
    print(f"Request completed in {elapsed_time:.2f} seconds")
    print(f"Status code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("\nRecommendations:")
        for idx, rec in enumerate(result.get('recommendations', []), 1):
            print(f"{idx}. {rec['title']} - Score: {rec['score']:.2f}%")
    else:
        print(f"Error: {response.text}")


if __name__ == "__main__":
    test_recommend_new_endpoint()
