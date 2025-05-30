import os
import torch

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Data paths
DATA_PATH = os.path.join(DATA_DIR, 'processed', 'filtered_IT_data.json')
ONET_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'onet_data.json')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'hetero_gnn_job_model.pt')
SYSTEM_PATH = os.path.join(BASE_DIR, 'models', 'mb', 'processed_system.pkl')
GRAPH_PATH = os.path.join(BASE_DIR, 'models', 'mb', 'hetero_graph_data.pt')
NEW_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mb', 'best_model1.pth')

# Model configuration
HIDDEN_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 2

# API configuration
ENABLE_DEBUG_LOGGING = True
DEFAULT_TOP_K = 3
MAX_SEARCH_RESULTS = 20

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
