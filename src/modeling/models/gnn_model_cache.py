"""
Cache implementation for the new GNN model data to avoid rebuilding the graph on every request
"""
import os
import json
import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, Optional

from src.config import DEVICE, NEW_MODEL_PATH, DATA_PATH
from src.utils.helpers import debug_log

# Make torch_geometric classes safe for loading
torch.serialization.add_safe_globals(
    ['torch_geometric.data.storage.BaseStorage'])
torch.serialization.add_safe_globals(['torch_geometric.data.data.Data'])
torch.serialization.add_safe_globals(
    ['torch_geometric.data.hetero_data.HeteroData'])
torch.serialization.add_safe_globals(
    ['torch_geometric.utils.undirected.to_undirected'])
torch.serialization.add_safe_globals(
    ['torch_geometric.data.storage.NodeStorage'])
torch.serialization.add_safe_globals(
    ['torch_geometric.data.storage.EdgeStorage'])

# Cache path constants
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(__file__)))), 'models', 'cache')
GRAPH_CACHE_PATH = os.path.join(CACHE_DIR, 'new_model_graph_data.pt')
EMBEDDINGS_CACHE_PATH = os.path.join(CACHE_DIR, 'new_model_embeddings.pt')
MAPPINGS_CACHE_PATH = os.path.join(CACHE_DIR, 'new_model_mappings.pkl')
SENTENCE_MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v4"

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)


class GNNModelCache:
    """
    Cache for the new GNN model data to avoid rebuilding the graph on every request
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GNNModelCache, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model = None
        self.data = None
        self.sentence_model = None
        self.job_id_to_title_map = {}
        self.tech_map = {}
        self.job_titles = []
        self._initialized = True

    def save_cache(self, data, mappings):
        """
        Save data to cache files
        """
        try:
            # Save graph data
            torch.save(data, GRAPH_CACHE_PATH)
            debug_log(f"Saved graph data cache to {GRAPH_CACHE_PATH}")

            # Save mappings
            with open(MAPPINGS_CACHE_PATH, 'wb') as f:
                pickle.dump(mappings, f)
            debug_log(f"Saved mappings cache to {MAPPINGS_CACHE_PATH}")

            return True
        except Exception as e:
            debug_log(f"Error saving cache data: {str(e)}")
            return False

    def load_cache(self):
        """
        Load data from cache files
        """
        try:
            # Check if cache files exist
            if not os.path.exists(GRAPH_CACHE_PATH) or not os.path.exists(MAPPINGS_CACHE_PATH):
                debug_log("Cache files don't exist")
                return False

            # Check if original data file has been modified
            if os.path.exists(DATA_PATH):
                data_mod_time = os.path.getmtime(DATA_PATH)
                cache_mod_time = os.path.getmtime(GRAPH_CACHE_PATH)

                if data_mod_time > cache_mod_time:
                    debug_log(
                        f"Data file {DATA_PATH} has been modified since cache was created")
                    return False

            # Load graph data
            self.data = torch.load(
                GRAPH_CACHE_PATH, map_location=DEVICE, weights_only=False)
            debug_log(f"Loaded graph data cache from {GRAPH_CACHE_PATH}")

            # Load mappings
            with open(MAPPINGS_CACHE_PATH, 'rb') as f:
                mappings = pickle.load(f)
                self.job_id_to_title_map = mappings['job_id_to_title_map']
                self.tech_map = mappings['tech_map']
                self.job_titles = mappings['job_titles']
            debug_log(f"Loaded mappings cache from {MAPPINGS_CACHE_PATH}")

            # Load model
            from src.services.api.routes import HeteroGNN

            # Get graph metadata and dimensions
            hidden_channels = 128
            gnn_out_channels = 128
            gnn_layers = 2
            sbert_dim = 768
            node_input_dims_map = {
                'job': sbert_dim,
                'technology': sbert_dim + 2,
                'skill_category': sbert_dim
            }

            # Create model
            self.model = HeteroGNN(
                sbert_dim=sbert_dim,
                hidden_channels=hidden_channels,
                out_channels=gnn_out_channels,
                num_layers=gnn_layers,
                metadata=self.data.metadata(),
                node_input_dims=node_input_dims_map
            )

            # Load model weights
            self.model.load_state_dict(torch.load(
                NEW_MODEL_PATH, map_location=DEVICE))
            self.model.to(DEVICE)
            self.model.eval()
            debug_log("Loaded and initialized model")

            # Load sentence transformer
            self.sentence_model = SentenceTransformer(
                SENTENCE_MODEL_NAME, device=DEVICE)
            debug_log("Loaded sentence transformer model")

            return True
        except Exception as e:
            debug_log(f"Error loading cache data: {str(e)}")
            return False


# Global cache instance
gnn_model_cache = GNNModelCache()
