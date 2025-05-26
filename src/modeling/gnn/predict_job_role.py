"""
Command-line tool for job role prediction using the Hybrid Ensemble model.
"""
import os
import sys
import argparse
import pandas as pd
from tabulate import tabulate
import textwrap

# Add parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import ensemble model
from modeling.hybrid_ensemble import HybridEnsembleClassifier

# Define project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

def predict_job_role(text=None, tech_file=None, top_k=3, model_dir=None, detailed=False):
    """
    Predict job roles from text or a list of technologies.
    
    Args:
        text: Input text to extract technologies from
        tech_file: File containing a list of technologies (one per line)
        top_k: Number of top predictions to return
        model_dir: Directory containing the model
        detailed: Whether to show detailed component predictions
        
    Returns:
        result: Dictionary with prediction results
    """
    # Load the model
    if model_dir is None:
        model_dir = os.path.join(MODEL_DIR, 'hybrid_ensemble')
    
    print(f"Loading model from {model_dir}...")
    ensemble = HybridEnsembleClassifier()
    ensemble.load_model(model_dir)
    print("Model loaded successfully")
    
    # Get technologies from text or file
    technologies = None
    
    if tech_file:
        # Read technologies from file
        with open(tech_file, 'r') as f:
            technologies = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Read {len(technologies)} technologies from {tech_file}")
    
    # Make prediction
    if text:
        result = ensemble.predict(input_text=text, top_k=top_k)
        print(f"Extracted {len(result['extracted_technologies'])} technologies from input text")
    elif technologies:
        result = ensemble.predict(technologies_list=technologies, top_k=top_k)
    else:
        raise ValueError("Either text or tech_file must be provided")
    
    # Display results
    print("\n--- TECHNOLOGIES ---")
    for tech in result["extracted_technologies"]:
        print(f"- {tech}")
    
    # Format predictions for display
    print("\n--- JOB ROLE PREDICTIONS ---")
    pred_table = []
    for i, pred in enumerate(result["ensemble_predictions"], 1):
        pred_table.append([
            i, 
            pred["role"], 
            f"{pred['probability']:.2%}"
        ])
    
    print(tabulate(
        pred_table, 
        headers=["Rank", "Job Role", "Confidence"],
        tablefmt="pretty"
    ))
    
    # Show detailed component predictions if requested
    if detailed:
        print("\n--- COMPONENT MODEL PREDICTIONS ---")
        comp_table = []
        
        # Extract unique job roles across all component models
        all_roles = {}
        
        # Add predictions from ML model
        if "ml" in result["component_predictions"]:
            for pred in result["component_predictions"]["ml"]["predictions"]:
                role = pred["role"]
                if role not in all_roles:
                    all_roles[role] = {"ML": 0, "NN": 0, "GNN": 0}
                all_roles[role]["ML"] = pred["probability"]
        
        # Add predictions from NN model
        if "nn" in result["component_predictions"]:
            for pred in result["component_predictions"]["nn"]["predictions"]:
                role = pred["role"]
                if role not in all_roles:
                    all_roles[role] = {"ML": 0, "NN": 0, "GNN": 0}
                all_roles[role]["NN"] = pred["probability"]
        
        # Add predictions from GNN model
        if "gnn" in result["component_predictions"]:
            for pred in result["component_predictions"]["gnn"]["predictions"]:
                role = pred["role"]
                if role not in all_roles:
                    all_roles[role] = {"ML": 0, "NN": 0, "GNN": 0}
                all_roles[role]["GNN"] = pred["probability"]
        
        # Format as table
        for role, scores in all_roles.items():
            comp_table.append([
                role,
                f"{scores['ML']:.2%}",
                f"{scores['NN']:.2%}",
                f"{scores['GNN']:.2%}"
            ])
        
        # Sort by ensemble prediction order
        ensemble_order = [pred["role"] for pred in result["ensemble_predictions"]]
        comp_table.sort(key=lambda x: ensemble_order.index(x[0]) if x[0] in ensemble_order else 999)
        
        # Display component model weights
        weights = result["ensemble_weights"]
        print(f"Model Weights: ML={weights['ml']:.2f}, NN={weights['nn']:.2f}, GNN={weights['gnn']:.2f}")
        
        # Display table
        print(tabulate(
            comp_table[:top_k+2],  # Show a few more than top_k
            headers=["Job Role", "ML Score", "NN Score", "GNN Score"],
            tablefmt="pretty"
        ))
    
    return result

def main():
    """
    Main function to parse arguments and make predictions.
    """
    parser = argparse.ArgumentParser(
        description='Predict job roles from text or a list of technologies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
        Examples:
          python predict_job_role.py --text "I am a developer with experience in Python, JavaScript, and React."
          python predict_job_role.py --tech-file my_skills.txt
          python predict_job_role.py --text "Full-stack developer with MongoDB, Express, React, and Node.js experience" --detailed
        ''')
    )
    
    parser.add_argument('--text', type=str,
                        help='Text to extract technologies from')
    
    parser.add_argument('--tech-file', type=str,
                        help='File containing a list of technologies (one per line)')
    
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of top predictions to return')
    
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Directory containing the model')
    
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed component predictions')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.text and not args.tech_file:
        parser.error("Either --text or --tech-file must be provided")
    
    if args.tech_file and not os.path.exists(args.tech_file):
        parser.error(f"Technology file not found: {args.tech_file}")
    
    # Make prediction
    try:
        predict_job_role(
            text=args.text,
            tech_file=args.tech_file,
            top_k=args.top_k,
            model_dir=args.model_dir,
            detailed=args.detailed
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
