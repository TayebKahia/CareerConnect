"""
Training script for the Dual Ensemble model (ML + NN).
This script trains and evaluates the Dual Ensemble model.
"""
import os
import sys
import argparse

# Add parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the dual ensemble classifier
from modeling.dual_ensemble import DualEnsembleClassifier

# Define project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')


def train_dual_ensemble(data_path=None, output_dir=None, force=False):
    """
    Train the Dual Ensemble model.
    
    Args:
        data_path: Path to the training data CSV
        output_dir: Directory to save the model in
        force: Whether to force training even if model exists
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Set default paths
    if data_path is None:
        data_path = os.path.join(DATA_PROCESSED_DIR, 'clean_v3.csv')
    
    if output_dir is None:
        output_dir = os.path.join(MODEL_DIR, 'dual_ensemble')
    
    # Check if model already exists
    if os.path.exists(output_dir) and not force:
        print(f"Model directory already exists: {output_dir}")
        print("Use --force to retrain the model")
        return None
    
    print(f"Training Dual Ensemble model...")
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    
    # Create and train ensemble
    ensemble = DualEnsembleClassifier()
    metrics = ensemble.train_models(data_path)
    
    # Save the model
    ensemble.save_model(output_dir)
    
    print(f"\nModel saved to {output_dir}")
    
    return metrics


def main():
    """
    Main function to parse arguments and train the model.
    """
    parser = argparse.ArgumentParser(
        description='Train the Dual Ensemble model (ML + NN)'
    )
    
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to the training data CSV')
    
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the model in')
    
    parser.add_argument('--force', action='store_true',
                        help='Force training even if model exists')
    
    args = parser.parse_args()
    
    # Train the model
    metrics = train_dual_ensemble(
        data_path=args.data_path,
        output_dir=args.output_dir,
        force=args.force
    )


if __name__ == "__main__":
    main()
