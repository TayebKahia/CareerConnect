import requests
import os
import sys
import json
from pprint import pprint


def test_cv_file_upload(pdf_path, api_url='http://localhost:5000', top_n=5):
    """Test CV-based job recommendation endpoint with a direct PDF file upload."""

    # Set API endpoint
    url = f"{api_url}/api/recommend-GNN-Onet-from-cv"

    print(f"Uploading PDF from: {pdf_path}")

    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return None

    # Make sure it's a PDF
    if not pdf_path.lower().endswith('.pdf'):
        print("Error: File must be a PDF")
        return None

    # Prepare files and data for multipart form upload
    files = {'file': (os.path.basename(pdf_path), open(
        pdf_path, 'rb'), 'application/pdf')}
    data = {'top_n': str(top_n)}

    print(f"Sending request to: {url}")

    # Send POST request
    try:
        response = requests.post(url, files=files, data=data)

        try:
            # Try to parse JSON response
            result = response.json()
        except json.JSONDecodeError:
            print("Error: Could not parse JSON response")
            print(f"Response status code: {response.status_code}")
            # Print first 500 chars
            print(f"Response text: {response.text[:500]}...")
            return None

        if response.status_code >= 400:
            print(f"Error: API returned status {response.status_code}")
            print(f"Error message: {result.get('message', 'Unknown error')}")
            return None

        # Print results
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
        return None
    finally:
        # Make sure to close the file
        files['file'][1].close()


if __name__ == "__main__":
    # Get arguments
    if len(sys.argv) < 2:
        print(
            "Usage: python test_cv_upload.py path_to_resume.pdf [api_url] [top_n]")
        print("Example: python test_cv_upload.py resume.pdf http://localhost:5000 5")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Optional API URL parameter
    api_url = "http://localhost:5000"
    if len(sys.argv) > 2:
        api_url = sys.argv[2]

    # Optional top_n parameter
    top_n = 5
    if len(sys.argv) > 3:
        try:
            top_n = int(sys.argv[3])
        except ValueError:
            print(f"Invalid top_n value: {sys.argv[3]}. Using default: 5")

    # Run the test
    test_cv_file_upload(pdf_path, api_url, top_n)
