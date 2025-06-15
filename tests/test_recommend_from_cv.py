from src.config import API_HOST, API_PORT
import requests
import base64
import json
import os
import sys
from pprint import pprint

# Add the parent directory to the path so we can import the configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_cv_recommendation(pdf_path, top_n=5):
    """Test CV-based job recommendation endpoint with a local PDF file."""

    # Set API endpoint
    url = f'http://{API_HOST}:{API_PORT}/api/recommend-GNN-Onet-from-cv'

    print(f"Reading PDF from: {pdf_path}")

    # Read PDF file and encode it in base64
    try:
        with open(pdf_path, 'rb') as file:
            pdf_content = file.read()
        pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return

    # Prepare request payload
    payload = {
        "pdf_base64": pdf_base64,
        "top_n": top_n
    }

    print(f"Sending request to: {url}")

    # Send POST request
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses

        result = response.json()

        print("\nAPI Response:")
        print(f"Status: {result.get('status')}")
        print(f"Request ID: {result.get('request_id')}")
        print(f"Total recommendations: {result.get('total_recommendations')}")
        print(f"Total extracted skills: {result.get('total_skills')}")

        print("\nExtracted Skills:")
        skills = result.get('extracted_skills', [])
        for i, skill in enumerate(skills[:10], start=1):  # Show first 10 skills
            print(
                f"{i}. {skill.get('name')} (similarity: {skill.get('similarity'):.2f})")

        if len(skills) > 10:
            print(f"... and {len(skills) - 10} more skills")

        print("\nJob Recommendations:")
        recommendations = result.get('recommendations', [])
        for i, job in enumerate(recommendations, start=1):
            print(f"{i}. {job.get('title')} (score: {job.get('raw_score'):.4f})")

        return result

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None


if __name__ == "__main__":
    # Replace with the path to your CV PDF file
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        print("Please provide a path to a PDF resume file")
        print(
            "Usage: python test_recommend_from_cv.py path_to_resume.pdf [top_n]")
        sys.exit(1)

    # Get optional top_n parameter
    top_n = 5
    if len(sys.argv) > 2:
        try:
            top_n = int(sys.argv[2])
        except ValueError:
            print(f"Invalid top_n value: {sys.argv[2]}. Using default: 5")

    # Run the test
    test_cv_recommendation(pdf_path, top_n)
