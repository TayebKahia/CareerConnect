"""
Training script for the Hybrid Ensemble job role prediction model.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import ensemble model
from modeling.hybrid_ensemble import HybridEnsembleClassifier

# Define project root - handles running from both src/ and src/modeling/ directories
try:
    # First try the standard path (when running from src/)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    if not os.path.exists(DATA_PROCESSED_DIR):
        # If that doesn't exist, try going up one more level (when running from src/modeling/)
        PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
        DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
except:
    # Fallback to a simple definition
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
    DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

def train_ensemble(data_path, output_dir=None, eval_only=False, weights=None):
    """
    Train the Hybrid Ensemble model.
    
    Args:
        data_path: Path to the training data CSV
        output_dir: Directory to save the model in (optional)
        eval_only: If True, only evaluate existing models without training
        weights: Custom ensemble weights for ML, NN, and GNN components
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print(f"{'Evaluating' if eval_only else 'Training'} Hybrid Ensemble model...")
    
    # Initialize ensemble model
    ensemble = HybridEnsembleClassifier(
        ensemble_weights=weights,
        concept_matcher_path=os.path.join(DATA_PROCESSED_DIR, 'technologies_with_abbreviations.csv')
    )
    
    # Train or load models
    if eval_only:
        # Load existing models
        model_path = output_dir if output_dir else os.path.join(MODEL_DIR, 'hybrid_ensemble')
        ensemble.load_model(model_path)
        print(f"Loaded ensemble model from {model_path}")
        
        # Extract metrics from model metadata
        metrics = {
            'ensemble_weights': ensemble.ensemble_weights
        }
    else:
        # Train models and get metrics
        metrics = ensemble.train_models(data_path)
        
        # Save the trained model
        if output_dir:
            ensemble.save_model(output_dir)
        else:
            ensemble.save_model()
    
    # Load a subset of data for evaluation
    data = pd.read_csv(data_path)
    test_data = data.sample(min(1000, len(data)), random_state=42)
    
    # Evaluate on test data
    evaluate_ensemble(ensemble, test_data)
    
    return metrics

def evaluate_ensemble(ensemble, test_data):
    """
    Evaluate the ensemble model on a test dataset.
    
    Args:
        ensemble: Trained HybridEnsembleClassifier instance
        test_data: DataFrame containing test data
    """
    print("\nEvaluating ensemble on test data...")
    
    # Extract DevTypes (true labels)
    true_labels = test_data['DevType'].values
    
    # Prepare predictions array
    predictions = []
    
    # Loop through each row in the test data
    for _, row in test_data.iterrows():
        # Extract technologies from each column
        tech_list = []
        for col in ensemble.ml_model.mlb:
            if col in test_data.columns and pd.notna(row[col]):
                for tech in row[col].split(';'):
                    tech = tech.strip()
                    if tech:
                        tech_list.append(tech)
        
        # Skip if no technologies found
        if not tech_list:
            continue
        
        # Make prediction
        try:
            result = ensemble.predict(technologies_list=tech_list, top_k=1)
            top_pred = result["ensemble_predictions"][0]["role"]
            predictions.append(top_pred)
        except Exception as e:
            print(f"Error predicting: {e}")
            # Use most common class as fallback
            predictions.append(true_labels[0])
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels[:len(predictions)], predictions)
    macro_f1 = f1_score(true_labels[:len(predictions)], predictions, average='macro')
    weighted_f1 = f1_score(true_labels[:len(predictions)], predictions, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")
    print(f"Test Weighted F1: {weighted_f1:.4f}")
    
    # Plot confusion matrix
    try:
        plt.figure(figsize=(14, 10))
        cm = confusion_matrix(true_labels[:len(predictions)], predictions)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=ensemble.label_encoder.classes_,
                   yticklabels=ensemble.label_encoder.classes_)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        cm_plot_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
        plt.savefig(cm_plot_path)
        print(f"Confusion matrix saved to {cm_plot_path}")
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

def main():
    """
    Main function to parse arguments and train/evaluate the model.
    """
    parser = argparse.ArgumentParser(description='Train or evaluate the Hybrid Ensemble model')
    
    parser.add_argument('--data', type=str, default=os.path.join(DATA_PROCESSED_DIR, 'clean_v3.csv'),
                        help='Path to the training data CSV')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Directory to save the model in')
    
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate existing models without training')
    
    parser.add_argument('--weights', type=float, nargs=3, default=None,
                        help='Custom ensemble weights for ML, NN, and GNN components')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        return
    
    # Train or evaluate model
    train_ensemble(
        data_path=args.data,
        output_dir=args.output,
        eval_only=args.eval_only,
        weights=args.weights
    )

if __name__ == "__main__":
    main()
