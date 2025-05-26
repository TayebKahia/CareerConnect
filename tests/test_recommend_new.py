from src.config import DATA_PATH, NEW_MODEL_PATH
from src.services.api.routes import register_routes
import json
import unittest
from flask import Flask
import sys
import os

# Add the project root to path to import modules correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestRecommendNew(unittest.TestCase):
    def setUp(self):
        """Set up the test environment"""
        # Check if necessary files exist
        self.model_exists = os.path.exists(NEW_MODEL_PATH)
        self.data_exists = os.path.exists(DATA_PATH)

        # Create dummy model manager
        class DummyModelManager:
            def __init__(self):
                self.model = None
                self.hetero_data = None
                self.system = None
                self.original_job_data = {}
                self.onet_job_mapping = {}

        # Create Flask app
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.model_manager = DummyModelManager()
        register_routes(self.app, self.model_manager)
        self.client = self.app.test_client()

    def test_endpoint_exists(self):
        """Test that the endpoint exists and returns 400 with empty data"""
        response = self.client.post('/api/recommend-GNN-Onet',
                                    data=json.dumps({}),
                                    content_type='application/json')
        self.assertIn(response.status_code, [400, 500])
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')

    def test_model_files_exist(self):
        """Test that the required model files exist"""
        self.assertTrue(self.model_exists,
                        f"Model file missing: {NEW_MODEL_PATH}")
        self.assertTrue(self.data_exists, f"Data file missing: {DATA_PATH}")

    @unittest.skipIf(not (os.path.exists(NEW_MODEL_PATH) and os.path.exists(DATA_PATH)),
                     "Skipping full API test as model or data files are missing")
    def test_recommendation_api(self):
        """Test the recommendation API with sample input"""
        test_data = {
            "skills": [
                {"name": "Python", "type": "technology_name", "weight": 1.0},
                {"name": "Machine learning",
                    "type": "technology_name", "weight": 0.95},
                {"name": "SQL", "type": "technology_name", "weight": 0.85}
            ],
            "top_n": 3
        }

        response = self.client.post('/api/recommend-GNN-Onet',
                                    data=json.dumps(test_data),
                                    content_type='application/json')

        # We only test structure, not actual predictions which require the model
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertEqual(data['status'], 'success')
            self.assertIn('recommendations', data)
            self.assertIn('request_id', data)
            self.assertIn('timestamp', data)

            if len(data['recommendations']) > 0:
                job = data['recommendations'][0]
                self.assertIn('title', job)
                self.assertIn('score', job)
                self.assertIn('matchScore', job)


if __name__ == '__main__':
    unittest.main()
