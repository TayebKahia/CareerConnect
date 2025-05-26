"""
Dual Ensemble model for job role prediction that combines ML and NN approaches.
This is a simplified version of the hybrid ensemble that omits the GNN component.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.preprocessing import LabelEncoder

# Add parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import component models
from modeling.ml_job_classifier import MLJobClassifier
from modeling.nn_job_classifier import NNJobClassifier
from features import ConceptMatcher

# Define project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Technology columns in the dataset
TECH_COLUMNS = [
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "PlatformHaveWorkedWith",
    "WebframeHaveWorkedWith", 
    "MiscTechHaveWorkedWith",
    "ToolsTechHaveWorkedWith",
    "EmbeddedHaveWorkedWith"
]


class DualEnsembleClassifier:
    """
    Dual Ensemble classifier that combines predictions from ML and NN models.
    """
    
    def __init__(self, 
                ml_model_path=None, 
                nn_model_path=None,
                ensemble_weights=None,
                concept_matcher_path=None):
        """
        Initialize the ensemble classifier.
        
        Args:
            ml_model_path: Path to the ML model directory
            nn_model_path: Path to the NN model directory
            ensemble_weights: Weights for the ensemble (ML, NN)
            concept_matcher_path: Path to the ConceptMatcher resources
        """
        # Default ensemble weights if not provided
        if ensemble_weights is None:
            self.ensemble_weights = np.array([0.5, 0.5])  # ML, NN
        else:
            self.ensemble_weights = np.array(ensemble_weights)
            # Normalize weights to sum to 1
            self.ensemble_weights = self.ensemble_weights / self.ensemble_weights.sum()
        
        # Initialize component models
        self.ml_model = None
        self.nn_model = None
        self.concept_matcher = None
        self.label_encoder = None
        
        # Load models if paths are provided
        if ml_model_path:
            self.ml_model = MLJobClassifier(model_path=ml_model_path)
            
        if nn_model_path:
            self.nn_model = NNJobClassifier(model_path=nn_model_path)
            
        # Initialize concept matcher
        self._initialize_concept_matcher(concept_matcher_path)
        
        # Ensure label encoder consistency
        self._align_label_encoders()
    
    def _initialize_concept_matcher(self, concept_matcher_path=None):
        """
        Initialize the ConceptMatcher for technology extraction.
        
        Args:
            concept_matcher_path: Path to ConceptMatcher resources
        """
        # Always use the correct data path from project root
        if concept_matcher_path is None:
            concept_matcher_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'technologies_with_abbreviations.csv')
        print(f"Loading concept matcher from: {concept_matcher_path}")
        
        # Initialize ConceptMatcher
        self.concept_matcher = ConceptMatcher(csv_path=concept_matcher_path)
        self.concept_matcher.load_concepts()
        self.concept_matcher.generate_concept_embeddings(load_if_exists=True)
        
        print("ConceptMatcher initialized successfully")
    
    def _align_label_encoders(self):
        """
        Ensure consistent label encoding across all models.
        Uses the label encoder from the first available model.
        """
        if self.ml_model is not None and self.ml_model.label_encoder is not None:
            self.label_encoder = self.ml_model.label_encoder
        elif self.nn_model is not None and self.nn_model.label_encoder is not None:
            self.label_encoder = self.nn_model.label_encoder
    
    def train_models(self, data_path):
        """
        Train all component models of the ensemble.
        
        Args:
            data_path: Path to the training data CSV
            
        Returns:
            metrics: Dictionary of evaluation metrics for all models
        """
        metrics = {}
        
        print("Training ML classifier...")
        if self.ml_model is None:
            self.ml_model = MLJobClassifier()
        ml_metrics = self.ml_model.train(data_path)
        metrics['ml'] = ml_metrics
        
        print("\nTraining NN classifier...")
        if self.nn_model is None:
            self.nn_model = NNJobClassifier()
        nn_metrics = self.nn_model.train(data_path)
        metrics['nn'] = nn_metrics
        
        # Update label encoder for consistency
        self._align_label_encoders()
        
        # Determine optimal ensemble weights based on validation performance
        self._optimize_weights(metrics)
        
        return metrics
    
    def _optimize_weights(self, metrics):
        """
        Optimize ensemble weights based on model performance.
        
        Args:
            metrics: Dictionary of evaluation metrics for all models
        """
        # Extract accuracy or F1 scores from metrics
        scores = []
        
        if 'ml' in metrics:
            scores.append(metrics['ml'].get('macro_f1', metrics['ml'].get('accuracy', 0.5)))
        else:
            scores.append(0.5)  # Default weight if model not trained
            
        if 'nn' in metrics:
            scores.append(metrics['nn'].get('macro_f1', metrics['nn'].get('accuracy', 0.5)))
        else:
            scores.append(0.5)
        
        # Convert scores to weights (better performance = higher weight)
        weights = np.array(scores)
        
        # Apply softmax to get normalized weights
        weights = np.exp(weights * 2)  # Multiply by 2 to emphasize differences
        weights = weights / weights.sum()
        
        self.ensemble_weights = weights
        print(f"Optimized ensemble weights: ML={weights[0]:.3f}, NN={weights[1]:.3f}")
    
    def extract_technologies(self, text):
        """
        Extract technologies from input text using ConceptMatcher.
        
        Args:
            text: Input text to extract technologies from
            
        Returns:
            technologies: List of extracted technology names
        """
        if self.concept_matcher is None:
            raise ValueError("ConceptMatcher not initialized. Cannot extract technologies.")
        
        # Process text to extract technologies
        tech_results = self.concept_matcher.process_text(text)
        technologies = [tech["name"] for tech in tech_results]
        
        return technologies
    
    def predict(self, input_text=None, technologies_list=None, top_k=3):
        """
        Predict job roles using the ensemble of models.
        
        Args:
            input_text: Raw text input (resume, job description, etc.)
            technologies_list: List of technology names (if already extracted)
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and component model results        """
        # Check if we have at least one model
        if not self.is_model_ready():
            raise ValueError("No models available for prediction. Train or load models first.")
        
        # Extract technologies if text is provided and technologies are not
        if technologies_list is None:
            if input_text is None:
                raise ValueError("Either input_text or technologies_list must be provided.")
            technologies_list = self.extract_technologies(input_text)
        
        # Make predictions with each available model
        predictions = {}
        role_scores = {}
        
        # Collect predictions from each model
        if self.ml_model is not None:
            ml_preds = self.ml_model.predict(technologies_list, top_k=len(self.label_encoder.classes_))
            predictions['ml'] = ml_preds
        
        if self.nn_model is not None:
            nn_preds = self.nn_model.predict(technologies_list, top_k=len(self.label_encoder.classes_))
            predictions['nn'] = nn_preds
        
        # Combine predictions using ensemble weights
        all_roles = set()
        for model_key, model_preds in predictions.items():
            for pred in model_preds['predictions']:
                all_roles.add(pred['role'])
        
        # Initialize scores for all roles
        for role in all_roles:
            role_scores[role] = 0.0
        
        # Apply weighted voting
        weight_idx = 0
        for model_key, model_preds in predictions.items():
            weight = self.ensemble_weights[weight_idx]
            weight_idx += 1
            
            for pred in model_preds['predictions']:
                role = pred['role']
                score = pred['probability']
                role_scores[role] += weight * score
        
        # Sort roles by score
        sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
        top_roles = sorted_roles[:top_k]
        
        # Format ensemble results
        ensemble_predictions = [
            {"role": role, "probability": float(score)} 
            for role, score in top_roles
        ]
        
        # Construct final result
        result = {
            "ensemble_predictions": ensemble_predictions,
            "extracted_technologies": technologies_list,
            "component_predictions": predictions,
            "ensemble_weights": {
                "ml": float(self.ensemble_weights[0]) if self.ml_model is not None else 0,
                "nn": float(self.ensemble_weights[1]) if self.nn_model is not None else 0
            }
        }
        
        return result
    
    def save_model(self, output_dir=None):
        """
        Save the ensemble model and its components.
        
        Args:
            output_dir: Directory to save the model in
        """
        if output_dir is None:
            output_dir = os.path.join(MODEL_DIR, 'dual_ensemble')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save component models
        if self.ml_model is not None:
            ml_dir = os.path.join(output_dir, 'ml_model')
            self.ml_model.save_model(ml_dir)
        
        if self.nn_model is not None:
            nn_dir = os.path.join(output_dir, 'nn_model')
            self.nn_model.save_model(nn_dir)
        
        # Save ensemble configuration
        ensemble_config = {
            'ensemble_weights': self.ensemble_weights,
            'label_encoder': self.label_encoder
        }
        
        config_path = os.path.join(output_dir, 'ensemble_config.joblib')
        joblib.dump(ensemble_config, config_path)
        
        print(f"Ensemble model and components saved to {output_dir}")
        
    def load_model(self, input_dir=None):
        """
        Load the ensemble model and its components.
        
        Args:
            input_dir: Directory containing the saved model
        """
        if input_dir is None:
            # Default path in src/models
            input_dir = os.path.join(MODEL_DIR, 'dual_ensemble')
            
            # If the default path doesn't exist, try the project root models directory
            if not os.path.exists(input_dir):
                # Try path relative to project root instead of src directory
                project_root_dir = os.path.dirname(PROJECT_ROOT)
                alt_dir = os.path.join(project_root_dir, 'models', 'dual_ensemble')
                if os.path.exists(alt_dir):
                    input_dir = alt_dir
                    print(f"Using project root model path: {input_dir}")
        
        print(f"Attempting to load dual ensemble model from {input_dir}")
        
        # Check if directory exists
        if not os.path.exists(input_dir):
            print(f"Warning: Model directory {input_dir} does not exist")
            
            # If we haven't tried the project root path yet, try it now
            if not input_dir.endswith(os.path.join('models', 'dual_ensemble')):
                project_root_dir = os.path.dirname(PROJECT_ROOT)
                alt_dir = os.path.join(project_root_dir, 'models', 'dual_ensemble')
                print(f"Trying alternate path: {alt_dir}")
                
                if os.path.exists(alt_dir):
                    input_dir = alt_dir
                    print(f"Found models at alternate path: {input_dir}")
                else:
                    print(f"Warning: Alternate model directory {alt_dir} does not exist")
                    return False
            else:
                # We already tried the project root path and it didn't exist
                return False
        
        # Load component models
        ml_dir = os.path.join(input_dir, 'ml_model')
        nn_dir = os.path.join(input_dir, 'nn_model')
        
        # Load ML model
        if os.path.exists(ml_dir):
            try:
                self.ml_model = MLJobClassifier(model_path=ml_dir)
                print(f"ML model loaded successfully from {ml_dir}")
            except Exception as e:
                print(f"Error loading ML model: {e}")
                self.ml_model = None
        else:
            print(f"ML model directory not found at {ml_dir}")
            self.ml_model = None
        
        # Load NN model
        if os.path.exists(nn_dir):
            try:
                self.nn_model = NNJobClassifier(model_path=nn_dir)
                print(f"NN model loaded successfully from {nn_dir}")
            except Exception as e:
                print(f"Error loading NN model: {e}")
                self.nn_model = None
        else:
            print(f"NN model directory not found at {nn_dir}")
            self.nn_model = None
        
        # Load ensemble configuration
        config_path = os.path.join(input_dir, 'ensemble_config.joblib')
        if os.path.exists(config_path):
            try:
                ensemble_config = joblib.load(config_path)
                self.ensemble_weights = ensemble_config.get('ensemble_weights', self.ensemble_weights)
                self.label_encoder = ensemble_config.get('label_encoder')
                print(f"Ensemble configuration loaded from {config_path}")
            except Exception as e:
                print(f"Error loading ensemble configuration: {e}")
        
        # Initialize concept matcher if not already done
        if self.concept_matcher is None:
            self._initialize_concept_matcher()
        
        # Ensure label encoder consistency
        if self.label_encoder is None:
            self._align_label_encoders()
        
        # Diagnostic check
        if self.ml_model is None and self.nn_model is None:
            print("WARNING: No models were successfully loaded. Prediction will not work!")
            return False
        
        print(f"Ensemble model and components loaded from {input_dir}")
        return True

    def is_model_ready(self):
        """
        Check if the model is ready for prediction.
        
        Returns:
            bool: True if at least one model component is loaded
        """
        return self.ml_model is not None or self.nn_model is not None


if __name__ == "__main__":
    # Example usage:
    data_path = os.path.join(DATA_PROCESSED_DIR, 'clean_v3.csv')
    
    # Create and train ensemble
    ensemble = DualEnsembleClassifier()
    metrics = ensemble.train_models(data_path)
    
    # Save the model
    ensemble.save_model()
    
    # Example prediction
    input_text = """
    I am a software developer with 5 years of experience in web development.
    My primary skills include JavaScript, React, Node.js, and MongoDB.
    I have also worked with AWS for deployment and Docker for containerization.
    I'm familiar with Python and have used it for data analysis with Pandas.
    """
    
    # Predict job roles
    predictions = ensemble.predict(input_text=input_text, top_k=3)
    
    print("\nExtracted Technologies:")
    for tech in predictions["extracted_technologies"]:
        print(f"- {tech}")
    
    print("\nEnsemble Predictions:")
    for pred in predictions["ensemble_predictions"]:
        print(f"{pred['role']}: {pred['probability']:.4f}")
