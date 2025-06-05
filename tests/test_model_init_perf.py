# Performance optimization test for model initialization
import sys
import time
import torch
import gc
import numpy as np
from src.utils.helpers import debug_log
from models.mb.classes.gnn_model_cache import gnn_model_cache


def clear_model_cache():
    """Force clear the model cache and run garbage collection"""
    print("Clearing model cache...")
    gnn_model_cache.model = None
    gnn_model_cache.data = None
    gnn_model_cache.sentence_model = None
    gnn_model_cache.job_id_to_title_map = {}
    gnn_model_cache.tech_map = {}
    gnn_model_cache.job_titles = []

    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model cache cleared")


def test_model_init_performance():
    """Test the performance of model initialization"""
    from src.services.api.job_recommendation_service import JobRecommendationService

    # First run - should initialize from scratch or cache
    service = JobRecommendationService()
    print("\nTest 1: First model initialization")
    start_time = time.time()
    success = service.initialize_new_model()
    end_time = time.time()

    print(f"Model initialization {'succeeded' if success else 'failed'}")
    print(f"Initialization time: {end_time - start_time:.2f} seconds")

    if hasattr(gnn_model_cache, 'last_load_times') and gnn_model_cache.last_load_times['total_load']:
        print(
            f"Graph load: {gnn_model_cache.last_load_times['graph_load']:.2f} seconds")
        print(
            f"Mappings load: {gnn_model_cache.last_load_times['mappings_load']:.2f} seconds")
        print(
            f"Model creation: {gnn_model_cache.last_load_times['model_creation']:.2f} seconds")
        print(
            f"Model weights load: {gnn_model_cache.last_load_times['model_load']:.2f} seconds")
        print(
            f"Sentence transformer: {gnn_model_cache.last_load_times['sentence_model_load']:.2f} seconds")

    # Test a sample recommendation
    test_skills = [("Python", "technology_name", 1.0),
                   ("Machine Learning", "technology_name", 0.9)]

    pred_start = time.time()
    recommendations = service._get_recommendations(test_skills, 5)
    pred_time = time.time() - pred_start

    print(f"\nPrediction time: {pred_time:.2f} seconds")
    print(f"Got {len(recommendations)} recommendations")
    print(
        f"Top recommendation: {recommendations[0]['title']} (score: {recommendations[0]['score']:.1f})")

    # Test multiple initializations
    print("\nTest 2: Multiple initializations (should be instant)")
    runs = []
    for i in range(5):
        start_time = time.time()
        service.initialize_new_model()  # Should be almost instant now
        duration = time.time() - start_time
        runs.append(duration)
        print(f"  Run {i+1}: {duration:.4f} seconds")

    print(f"Average initialization time: {np.mean(runs):.4f} seconds")
    print(f"Median initialization time: {np.median(runs):.4f} seconds")

    # Clear cache and test again
    print("\nTest 3: Initialization after cache clearing")
    clear_model_cache()

    start_time = time.time()
    success = service.initialize_new_model()
    end_time = time.time()

    print(f"Model initialization {'succeeded' if success else 'failed'}")
    print(f"Initialization time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    print("Starting model initialization performance test...")
    test_model_init_performance()
