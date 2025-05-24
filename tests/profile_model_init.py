# Profile model initialization time
import time
import torch
from src.services.api.job_recommendation_service import JobRecommendationService
from models.mb.classes.gnn_model_cache import gnn_model_cache

# Clear any existing model data
gnn_model_cache.model = None
gnn_model_cache.data = None
gnn_model_cache.sentence_model = None
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Now time the model initialization
service = JobRecommendationService()

print("Starting model initialization...")
start_time = time.time()
success = service.initialize_new_model()
end_time = time.time()

print(
    f"Model initialization {'completed successfully' if success else 'failed'} in {end_time - start_time:.2f} seconds")

# Run a test prediction to make sure everything works
test_skills = [("Python", "technology_name", 1.0),
               ("Machine Learning", "technology_name", 0.9)]
print("Testing recommendation function...")
test_start = time.time()
recommendations = service._get_recommendations(test_skills, 5)
test_end = time.time()

print(f"Test prediction completed in {test_end - test_start:.2f} seconds")
print(f"Got {len(recommendations)} recommendations")
