"""
Unit tests for the Dual Ensemble model and job prediction functionality.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Add parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import modules to test
from src.modeling.dual_ensemble import DualEnsembleClassifier
from src.modeling.predict_dual import predict_job_role, DualJobRolePredictor
from src.features import ConceptMatcher

# Define project root and paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')


class TestDualEnsemble(unittest.TestCase):
    """Tests for the Dual Ensemble model."""
    
    @patch('src.modeling.dual_ensemble.MLJobClassifier')
    @patch('src.modeling.dual_ensemble.NNJobClassifier')
    @patch('src.modeling.dual_ensemble.ConceptMatcher')
    def setUp(self, mock_concept_matcher, mock_nn, mock_ml):
        """Set up tests with mocked components."""
        # Mock ConceptMatcher
        self.mock_concept_matcher_instance = mock_concept_matcher.return_value
        self.mock_concept_matcher_instance.process_text.return_value = [
            {"name": "Python", "similarity_score": 0.95},
            {"name": "JavaScript", "similarity_score": 0.93},
            {"name": "React", "similarity_score": 0.91}
        ]
        
        # Mock ML model
        self.mock_ml_instance = mock_ml.return_value
        self.mock_ml_instance.predict.return_value = {
            "predictions": [
                {"role": "Developer, full-stack", "probability": 0.7},
                {"role": "Developer, back-end", "probability": 0.2},
                {"role": "Data scientist", "probability": 0.1}
            ]
        }
        
        # Mock NN model
        self.mock_nn_instance = mock_nn.return_value
        self.mock_nn_instance.predict.return_value = {
            "predictions": [
                {"role": "Developer, front-end", "probability": 0.6},
                {"role": "Developer, full-stack", "probability": 0.3},
                {"role": "Developer, back-end", "probability": 0.1}
            ]
        }
        
        # Create ensemble
        self.ensemble = DualEnsembleClassifier(
            ensemble_weights=[0.5, 0.5]
        )
        
        # Set mocked components
        self.ensemble.ml_model = self.mock_ml_instance
        self.ensemble.nn_model = self.mock_nn_instance
        self.ensemble.concept_matcher = self.mock_concept_matcher_instance
    
    def test_extract_technologies(self):
        """Test technology extraction."""
        # Call the method
        technologies = self.ensemble.extract_technologies("Sample text")
        
        # Verify the result
        self.assertEqual(technologies, ["Python", "JavaScript", "React"])
        self.mock_concept_matcher_instance.process_text.assert_called_once_with("Sample text")
    
    def test_predict_with_text(self):
        """Test prediction with text input."""
        # Call the method
        result = self.ensemble.predict(input_text="Sample text")
        
        # Verify technology extraction was called
        self.mock_concept_matcher_instance.process_text.assert_called_once_with("Sample text")
        
        # Verify component models were called with extracted technologies
        self.mock_ml_instance.predict.assert_called_once()
        self.mock_nn_instance.predict.assert_called_once()
        
        # Verify ensemble predictions
        self.assertIn("ensemble_predictions", result)
        self.assertIn("extracted_technologies", result)
        self.assertEqual(result["extracted_technologies"], ["Python", "JavaScript", "React"])
        
        # Top prediction should be "Developer, full-stack" due to ensemble weights
        self.assertEqual(result["ensemble_predictions"][0]["role"], "Developer, full-stack")
    
    def test_predict_with_technologies(self):
        """Test prediction with technology list input."""
        # Call the method
        technologies = ["Python", "JavaScript", "React"]
        result = self.ensemble.predict(technologies_list=technologies)
        
        # Verify component models were called with the technologies
        self.mock_ml_instance.predict.assert_called_once_with(technologies, top_k=3)
        self.mock_nn_instance.predict.assert_called_once_with(technologies, top_k=3)
        
        # Verify ensemble predictions
        self.assertIn("ensemble_predictions", result)
        self.assertEqual(result["extracted_technologies"], technologies)
        
        # Top prediction should be "Developer, full-stack" due to ensemble weights
        self.assertEqual(result["ensemble_predictions"][0]["role"], "Developer, full-stack")


class TestDualJobRolePredictor(unittest.TestCase):
    """Tests for the DualJobRolePredictor interface."""
    
    @patch('src.modeling.predict_dual.DualEnsembleClassifier')
    def setUp(self, mock_ensemble_class):
        """Set up tests with mocked ensemble."""
        # Mock ensemble
        self.mock_ensemble = mock_ensemble_class.return_value
        self.mock_ensemble.predict.return_value = {
            "ensemble_predictions": [
                {"role": "Developer, full-stack", "probability": 0.5},
                {"role": "Developer, front-end", "probability": 0.3},
                {"role": "Developer, back-end", "probability": 0.2}
            ],
            "extracted_technologies": ["Python", "JavaScript", "React"]
        }
        
        self.mock_ensemble.extract_technologies.return_value = ["Python", "JavaScript", "React"]
        
        # Create predictor
        self.predictor = DualJobRolePredictor()
        self.predictor.ensemble = self.mock_ensemble
    
    def test_predict_from_text(self):
        """Test prediction from text."""
        # Call the method
        result = self.predictor.predict_from_text("Sample text")
        
        # Verify ensemble was called
        self.mock_ensemble.predict.assert_called_once_with(input_text="Sample text", top_k=3)
        
        # Verify result
        self.assertEqual(result["ensemble_predictions"][0]["role"], "Developer, full-stack")
    
    def test_predict_from_technologies(self):
        """Test prediction from technologies."""
        # Call the method
        technologies = ["Python", "JavaScript", "React"]
        result = self.predictor.predict_from_technologies(technologies)
        
        # Verify ensemble was called
        self.mock_ensemble.predict.assert_called_once_with(technologies_list=technologies, top_k=3)
        
        # Verify result
        self.assertEqual(result["ensemble_predictions"][0]["role"], "Developer, full-stack")
    
    def test_extract_technologies(self):
        """Test technology extraction."""
        # Call the method
        technologies = self.predictor.extract_technologies("Sample text")
        
        # Verify ensemble was called
        self.mock_ensemble.extract_technologies.assert_called_once_with("Sample text")
        
        # Verify result
        self.assertEqual(technologies, ["Python", "JavaScript", "React"])


if __name__ == '__main__':
    unittest.main()
