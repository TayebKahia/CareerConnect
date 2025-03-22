import requests
import json


def test_api(url="http://localhost:8000"):
    """
    Test the Job Recommendation API by sending sample requests
    """
    # Test the root endpoint
    print("Testing root endpoint...")
    response = requests.get(url)
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}\n")

    # Test getting technologies
    print("Testing technologies endpoint...")
    response = requests.get(f"{url}/skills/technologies")
    print(f"Status code: {response.status_code}")
    print(f"First 5 technologies: {response.json()[:5]}\n")

    # Test getting skill categories
    print("Testing skill categories endpoint...")
    response = requests.get(f"{url}/skills/categories")
    print(f"Status code: {response.status_code}")
    print(f"First 5 skill categories: {response.json()[:5]}\n")

    # Test job recommendation with Data Science skills
    print("Testing job recommendation with Data Science skills...")
    data_science_skills = {
        "skills": [
            {"name": "Python", "type": "technology_name", "weight": 1.0},
            {"name": "R", "type": "technology_name", "weight": 0.9},
            {"name": "SQL", "type": "technology_name", "weight": 0.85},
            {"name": "Machine learning", "type": "skill_title", "weight": 0.95},
            {"name": "Data analysis", "type": "skill_title", "weight": 1.0},
            {"name": "TensorFlow", "type": "technology_name", "weight": 0.8},
            {"name": "PyTorch", "type": "technology_name", "weight": 0.75}
        ],
        "top_n": 5
    }

    response = requests.post(
        f"{url}/recommend",
        data=json.dumps(data_science_skills),
        headers={'Content-Type': 'application/json'}
    )
    print(f"Status code: {response.status_code}")
    print("Recommendations:")
    if response.status_code == 200:
        for i, rec in enumerate(response.json()["recommendations"]):
            print(f"  {i+1}. {rec['title']} (Score: {rec['score']:.4f})")
    else:
        print(f"Error: {response.json()}")
    print()

    # Test job recommendation with Web Development skills
    print("Testing job recommendation with Web Development skills...")
    web_dev_skills = {
        "skills": [
            {"name": "JavaScript", "type": "technology_name", "weight": 1.0},
            {"name": "HTML", "type": "technology_name", "weight": 0.95},
            {"name": "CSS", "type": "technology_name", "weight": 0.9},
            {"name": "React", "type": "technology_name", "weight": 0.85},
            {"name": "Node.js", "type": "technology_name", "weight": 0.8},
            {"name": "Web development", "type": "skill_title", "weight": 1.0}
        ],
        "top_n": 5
    }

    response = requests.post(
        f"{url}/recommend",
        data=json.dumps(web_dev_skills),
        headers={'Content-Type': 'application/json'}
    )
    print(f"Status code: {response.status_code}")
    print("Recommendations:")
    if response.status_code == 200:
        for i, rec in enumerate(response.json()["recommendations"]):
            print(f"  {i+1}. {rec['title']} (Score: {rec['score']:.4f})")
    else:
        print(f"Error: {response.json()}")


if __name__ == "__main__":
    test_api()
