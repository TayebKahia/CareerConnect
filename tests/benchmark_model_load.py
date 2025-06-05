"""
Benchmark script to measure model initialization time
"""
import time
import torch
import gc
from src.utils.helpers import debug_log
from src.services.api.job_recommendation_service import JobRecommendationService
from models.mb.classes.gnn_model_cache import gnn_model_cache


def clear_cache():
    """Clear all cached data and force garbage collection"""
    gnn_model_cache.model = None
    gnn_model_cache.data = None
    gnn_model_cache.sentence_model = None
    gnn_model_cache.job_id_to_title_map = {}
    gnn_model_cache.tech_map = {}
    gnn_model_cache.job_titles = []
    gnn_model_cache._initialized = True

    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_model_init():
    """Benchmark and profile model initialization"""
    # Clear any existing cache first
    clear_cache()

    service = JobRecommendationService()

    # Benchmark cache loading
    debug_log("Benchmarking cache loading...")
    start_time = time.time()
    cache_success = gnn_model_cache.load_cache()
    cache_time = time.time() - start_time
    debug_log(
        f"Cache loading {'succeeded' if cache_success else 'failed'} in {cache_time:.4f} seconds")

    if not cache_success:
        # Clear again if cache loading failed
        clear_cache()

        # Benchmark full initialization
        debug_log("Benchmarking full model initialization...")
        start_time = time.time()

        # Measure data loading time
        data_load_start = time.time()
        data_load_time = time.time() - data_load_start
        debug_log(f"Data loading took {data_load_time:.4f} seconds")

        # Initialize model
        init_success = service.initialize_new_model()
        total_time = time.time() - start_time
        debug_log(
            f"Full initialization {'succeeded' if init_success else 'failed'} in {total_time:.4f} seconds")

    # Test a model prediction to ensure everything works
    debug_log("Testing model prediction...")
    test_skills = [("Python", "technology_name", 1.0),
                   ("Machine Learning", "technology_name", 0.9)]

    start_time = time.time()
    recommendations = service._get_recommendations(test_skills, 5)
    prediction_time = time.time() - start_time
    debug_log(f"Model prediction took {prediction_time:.4f} seconds")
    debug_log(f"Got {len(recommendations)} recommendations")

    return {
        "cache_loading_time": cache_time if 'cache_time' in locals() else None,
        "cache_loaded": cache_success if 'cache_success' in locals() else None,
        "full_init_time": total_time if 'total_time' in locals() else None,
        "data_load_time": data_load_time if 'data_load_time' in locals() else None,
        "prediction_time": prediction_time,
        "recommendation_count": len(recommendations)
    }


if __name__ == "__main__":
    debug_log("Starting model initialization benchmark...")
    results = benchmark_model_init()
    debug_log(f"Benchmark results: {results}")
