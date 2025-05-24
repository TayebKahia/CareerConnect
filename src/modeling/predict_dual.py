"""
Primary prediction interface for job role prediction using the Dual Ensemble model.
This module provides a simple interface to the Dual Ensemble model for job role prediction.
"""
import os
import sys

# Add parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import ensemble model and ConceptMatcher
from modeling.dual_ensemble import DualEnsembleClassifier
from features import ConceptMatcher

# Define project root and paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')


class DualJobRolePredictor:
    """
    Interface for predicting job roles from text or technology lists using the Dual Ensemble model.
    This class provides a simple interface to the Dual Ensemble model.
    """
    
    def __init__(self, model_dir=None, concept_matcher_path=None):
        """
        Initialize the job role predictor.
        
        Args:
            model_dir: Directory containing the Dual Ensemble model
            concept_matcher_path: Path to ConceptMatcher resources
        """        # Set default paths if not provided
        if model_dir is None:
            # Try the default path in src/models
            model_dir = os.path.join(MODEL_DIR, 'dual_ensemble')
            
            # If it doesn't exist, try the project root path
            if not os.path.exists(model_dir):
                project_root_dir = os.path.dirname(PROJECT_ROOT)
                alt_dir = os.path.join(project_root_dir, 'models', 'dual_ensemble')
                if os.path.exists(alt_dir):
                    model_dir = alt_dir
                    print(f"Using model directory from project root: {model_dir}")
        
        if concept_matcher_path is None:
            concept_matcher_path = os.path.join(DATA_PROCESSED_DIR, 'technologies_with_abbreviations.csv')
        
        # Initialize ensemble model
        self.ensemble = DualEnsembleClassifier(concept_matcher_path=concept_matcher_path)
        
        # Try to load the model
        load_success = self.ensemble.load_model(model_dir)
        
        if load_success:
            print(f"Loaded dual ensemble model from {model_dir}")
        else:
            print(f"Warning: Failed to load models from any path.")
            print("Model needs to be trained before predictions can be made")
    
    def predict_from_text(self, text, top_k=3):
        """
        Predict job roles from input text.
        
        Args:
            text: Input text to extract technologies from
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        return self.ensemble.predict(input_text=text, top_k=top_k)
    
    def predict_from_technologies(self, technologies, top_k=3):
        """
        Predict job roles from a list of technologies.
        
        Args:
            technologies: List of technology names
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        return self.ensemble.predict(technologies_list=technologies, top_k=top_k)
    
    def extract_technologies(self, text):
        """
        Extract technologies from input text.
        
        Args:
            text: Input text to extract technologies from
            
        Returns:
            List of extracted technology names
        """
        return self.ensemble.extract_technologies(text)


# Singleton instance for easy import
_predictor = None

def get_predictor(model_dir=None, concept_matcher_path=None):
    """
    Get the singleton predictor instance.
    
    Args:
        model_dir: Directory containing the Dual Ensemble model
        concept_matcher_path: Path to ConceptMatcher resources
        
    Returns:
        DualJobRolePredictor instance
    """
    global _predictor
    if _predictor is None:
        _predictor = DualJobRolePredictor(model_dir, concept_matcher_path)
    return _predictor


def predict_job_role(text=None, technologies=None, top_k=3):
    """
    Predict job roles from text or a list of technologies.
    
    Args:
        text: Input text to extract technologies from
        technologies: List of technology names
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with prediction results
    """
    predictor = get_predictor()
    
    if text is not None:
        return predictor.predict_from_text(text, top_k)
    elif technologies is not None:
        return predictor.predict_from_technologies(technologies, top_k)
    else:
        raise ValueError("Either text or technologies must be provided")


if __name__ == "__main__":
    # Example usage
    sample_text = """
    I am a software developer with experience in Python, JavaScript, and React.
    I have worked with MongoDB and AWS for cloud deployments.
    """
    
    result = predict_job_role(text=sample_text)
    
    print("Extracted Technologies:")
    for tech in result["extracted_technologies"]:
        print(f"- {tech}")
    
    print("\nJob Role Predictions:")
    for pred in result["ensemble_predictions"]:
        print(f"{pred['role']}: {pred['probability']:.4f}")
