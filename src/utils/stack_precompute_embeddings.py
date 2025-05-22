"""
Utility script to pre-compute embeddings for the ConceptMatcher.

Run this script once before deploying the API to generate embedding files.
Embeddings are saved to disk and later loaded by the API to speed up processing.

Usage:
    python precompute_embeddings.py [model_name] [csv_path]

Arguments:
    model_name - Name of the sentence transformer model (default: all-mpnet-base-v2)
    csv_path - Path to the CSV with technology data (default: ../data/processed/technologies_with_abbreviations.csv)

Examples:
    python precompute_embeddings.py
    python precompute_embeddings.py all-MiniLM-L6-v2
    python precompute_embeddings.py all-mpnet-base-v2 /path/to/technologies.csv
"""

import os
import sys
import time
import argparse

# Add the parent directory to sys.path to be able to import ConceptMatcher
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)

from src.features import ConceptMatcher


def precompute_embeddings(
    model_name="sentence-transformers/msmarco-distilbert-base-v4", csv_path=None
):
    """
    Pre-compute and save embeddings for the specified model and CSV path.

    Args:
        model_name (str): The sentence transformer model to use
        csv_path (str): Path to the CSV containing technology data
    """
    if csv_path is None:
        # Default path relative to the project root
        csv_path = os.path.join(
            parent_dir, "data/processed/technologies_with_abbreviations.csv"
        )

    print(f"Pre-computing embeddings for model: {model_name}")
    print(f"Using technology data from: {csv_path}")

    start_time = time.time()

    # Initialize the ConceptMatcher
    matcher = ConceptMatcher(
        csv_path=csv_path,
        model_name=model_name,
    )

    # Load concepts
    print("Loading technology concepts...")
    matcher.load_concepts()

    # Generate and save embeddings
    print("Generating embeddings (this may take a while)...")
    matcher.generate_concept_embeddings(save_embeddings=True, load_if_exists=False)

    processing_time = time.time() - start_time

    print(f"Embeddings successfully generated and saved!")
    print(f"Processing time: {processing_time:.2f} seconds")

    # Get the embedding file paths for reference
    filename_notebook = (
        f"notebooks/stack_concept_embeddings_{model_name.replace('/', '_')}.npy"
    )
    filename_current = f"stack_concept_embeddings_{model_name.replace('/', '_')}.npy"

    if os.path.exists(filename_notebook):
        print(f"Embeddings saved to: {os.path.abspath(filename_notebook)}")
    elif os.path.exists(filename_current):
        print(f"Embeddings saved to: {os.path.abspath(filename_current)}")
    else:
        print(
            "Warning: Embedding file not found after generation. Please check for errors."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute embeddings for the ConceptMatcher"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="all-mpnet-base-v2",
        help="Name of the sentence transformer model",
    )
    parser.add_argument(
        "csv", nargs="?", default=None, help="Path to the CSV with technology data"
    )

    args = parser.parse_args()

    precompute_embeddings(args.model, args.csv)
