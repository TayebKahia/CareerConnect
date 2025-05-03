import requests
import json
import time

BASE_URL = 'http://localhost:5000'

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n=== Testing Root Endpoint ===")
    response = requests.get(BASE_URL)
    print(f"Status Code: {response.status_code}")
    if response.ok:
        data = response.json()
        print("Available Endpoints:")
        for endpoint in data.get('available_endpoints', []):
            print(f"- {endpoint['path']} ({endpoint['method']})")

def test_skills_endpoint():
    """Test the /api/skills endpoint"""
    print("\n=== Testing /api/skills ===")
    response = requests.get(f'{BASE_URL}/api/skills')
    print(f"Status Code: {response.status_code}")
    if response.ok:
        data = response.json()
        print(f"Total Skills: {data.get('total_skills', 0)}")
        print(f"Total Technologies: {data.get('total_technologies', 0)}")
        print("\nExample Skills:", data.get('skills', [])[:3])
        print("Example Technologies:", data.get('technologies', [])[:3])

def test_skill_search():
    """Test the /api/skill-search endpoint"""
    query = "python"
    print(f"\n=== Testing /api/skill-search with query '{query}' ===")
    response = requests.get(f'{BASE_URL}/api/skill-search', params={'q': query})
    print(f"Status Code: {response.status_code}")
    if response.ok:
        data = response.json()
        print(f"Total Matches: {data.get('total_matches', 0)}")
        print("Matching Skills:", data.get('skills', []))
        print("Matching Technologies:", data.get('technologies', []))

def test_recommendations():
    """Test the /api/recommend endpoint"""
    print("\n=== Testing /api/recommend ===")
    payload = {
        'skills': [
            {'name': 'Python', 'type': 'technology', 'similarity': 1.0},
            {'name': 'Machine Learning', 'type': 'skill', 'similarity': 0.95},
            {'name': 'Data Analysis', 'type': 'skill', 'similarity': 1.0}
        ],
        'top_k': 3
    }
    
    headers = {'Content-Type': 'application/json'}
    response = requests.post(
        f'{BASE_URL}/api/recommend',
        json=payload,
        headers=headers
    )
    
    print(f"Status Code: {response.status_code}")
    if response.ok:
        data = response.json()
        print(f"Request ID: {data.get('request_id')}")
        print(f"Total Recommendations: {data.get('total_recommendations', 0)}")
        
        for i, job in enumerate(data.get('recommendations', []), 1):
            print(f"\nRecommendation {i}:")
            print(f"Title: {job.get('title')}")
            print(f"Confidence: {job.get('confidence'):.2f}%")
            print(f"Demand: {job.get('demand_percentage', 0):.2f}%")
            print(f"Hot Tech: {job.get('hot_tech_percentage', 0):.2f}%")

def main():
    """Run all tests"""
    print("Starting API Tests...")
    try:
        # Test all endpoints
        test_root_endpoint()
        test_skills_endpoint()
        test_skill_search()
        test_recommendations()
        print("\nAll tests completed!")
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:5000")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")

if __name__ == '__main__':
    main() 