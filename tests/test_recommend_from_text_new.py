"""
Test script to validate the new text-based recommendation API using the newer model
"""
import requests
import json
import time


def test_recommend_from_text_new_endpoint():
    """Test the /api/recommend-GNN-Onet-from-text endpoint"""
    url = "http://localhost:5000/api/recommend-GNN-Onet-from-text"

    # Test payload with sample text
    payload = {
        "text": """I am a data scientist with expertise in Python and machine learning. 
        I have experience with TensorFlow, PyTorch, and SQL databases. 
        I've worked on data visualization projects using visualization libraries 
        and have some knowledge of web frameworks. I'm familiar with cloud 
        platforms like AWS and have worked with PaaS solutions.""",
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
            print(
                f"{idx}. {rec['title']} - Raw Score: {rec['matchScore']:.6f}")

        print("\nExtracted Skills:")
        for idx, skill in enumerate(result.get('extracted_skills', []), 1):
            print(
                f"{idx}. {skill['name']} - Similarity: {skill['similarity']:.2f}")
    else:
        print(f"Error: {response.text}")


if __name__ == "__main__":
    test_recommend_from_text_new_endpoint()
