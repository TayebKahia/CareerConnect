from src.modeling.models.hetero_gnn_recommendation import (
    HeteroJobRecommendationSystem,
    HGNNJobRecommender
)
from src.utils.helpers import debug_log
from src.config import *
import os
import json
import torch
import pickle
from sentence_transformers import SentenceTransformer
from typing import Tuple, Dict, Any, Optional

# Add safe globals for PyTorch Geometric classes
import torch_geometric
from torch.serialization import add_safe_globals

# Make torch_geometric classes safe for loading
add_safe_globals(['torch_geometric.data.storage.BaseStorage'])
add_safe_globals(['torch_geometric.data.data.Data'])
add_safe_globals(['torch_geometric.data.hetero_data.HeteroData'])
add_safe_globals(['torch_geometric.utils.undirected.to_undirected'])
add_safe_globals(['torch_geometric.data.storage.NodeStorage'])
add_safe_globals(['torch_geometric.data.storage.EdgeStorage'])


class ModelManager:
    def __init__(self):
        self.model = None
        self.hetero_data = None
        self.system = None
        self.sentence_transformer = None
        self.original_job_data = None
        self.onet_job_data = None
        self.onet_job_mapping = {}

    def load_job_data(self) -> bool:
        """Load the original job data with complete details"""
        try:
            debug_log(f"Loading job data from {DATA_PATH}...")
            with open(DATA_PATH, 'r') as f:
                data = json.load(f)

            # Create a title-to-job mapping
            job_data_by_title = {}
            for job in data:
                job_title = job.get('title')
                if job_title:
                    job_data_by_title[job_title] = job

            self.original_job_data = job_data_by_title
            debug_log(f"Loaded details for {len(self.original_job_data)} jobs")
            return True
        except Exception as e:
            debug_log(f"Error loading job data: {str(e)}")
            return False

    def load_onet_data(self) -> bool:
        """Load O*NET job data"""
        try:
            debug_log(f"Loading O*NET job data from {ONET_DATA_PATH}...")
            with open(ONET_DATA_PATH, 'r') as f:
                self.onet_job_data = json.load(f)

            # Create mappings for easier lookup
            for job in self.onet_job_data:
                if 'code' in job:
                    onet_code = job['code']
                    self.onet_job_mapping[onet_code] = job

                    if 'title' in job:
                        title = job['title'].lower()
                        self.onet_job_mapping[title] = job

            debug_log(f"Loaded {len(self.onet_job_data)} O*NET job entries")
            return True
        except Exception as e:
            debug_log(f"Error loading O*NET job data: {str(e)}")
            return False

    def save_processed_data(self) -> bool:
        """Save all processed data (system and graph) to permanent files"""
        try:
            debug_log(f"Saving processed system to {SYSTEM_PATH}...")
            with open(SYSTEM_PATH, 'wb') as f:
                pickle.dump(self.system, f)

            debug_log(f"Saving graph data to {GRAPH_PATH}...")
            torch.save(self.hetero_data, GRAPH_PATH)
            return True
        except Exception as e:
            debug_log(f"Error saving processed data: {str(e)}")
            return False

    def load_processed_data(self) -> bool:
        """Load the system and graph data from permanent files"""
        debug_log("Checking for saved processed data...")
        system_exists = os.path.exists(SYSTEM_PATH)
        graph_exists = os.path.exists(GRAPH_PATH)
        data_modified = False

        # Check if data file was modified
        if os.path.exists(DATA_PATH) and (system_exists or graph_exists):
            data_mod_time = os.path.getmtime(DATA_PATH)
            if system_exists:
                system_mod_time = os.path.getmtime(SYSTEM_PATH)
                if data_mod_time > system_mod_time:
                    data_modified = True
            if graph_exists:
                graph_mod_time = os.path.getmtime(GRAPH_PATH)
                if data_mod_time > graph_mod_time:
                    data_modified = True

        if data_modified:
            return False
        try:
            if system_exists and graph_exists:
                with open(SYSTEM_PATH, 'rb') as f:
                    self.system = pickle.load(f)
                # Load with weights_only=False to support torch_geometric data structures
                self.hetero_data = torch.load(
                    GRAPH_PATH, map_location=DEVICE, weights_only=False)
                return True
            return False
        except Exception as e:
            debug_log(f"Error loading processed data: {str(e)}")
            return False

    def initialize_model(self) -> bool:
        """Initialize and load the model and all necessary data"""
        try:
            # Load the original job data first
            if not self.load_job_data():
                return False

            # Load O*NET job data
            if not self.load_onet_data():
                return False

            # Try to load processed data first
            if self.load_processed_data():
                # Only need to load the sentence transformer
                debug_log("Loading Sentence Transformer model...")
                model_name = "sentence-transformers/msmarco-distilbert-base-v4"
                self.sentence_transformer = SentenceTransformer(model_name)
            else:
                # Process everything from scratch
                debug_log("Processing data from scratch...")
                with open(DATA_PATH, 'r') as f:
                    data = json.load(f)

                model_name = "sentence-transformers/msmarco-distilbert-base-v4"
                self.sentence_transformer = SentenceTransformer(model_name)

                self.system = HeteroJobRecommendationSystem(data)
                self.hetero_data = self.system.process_data()
                self.save_processed_data()

            # Initialize the model
            self.model = HGNNJobRecommender(
                self.hetero_data,
                hidden_dim=HIDDEN_DIM,
                num_heads=NUM_HEADS,
                num_layers=NUM_LAYERS
            ).to(DEVICE)

            # Load model weights if available
            if os.path.exists(MODEL_PATH):
                self.model.load_state_dict(
                    torch.load(MODEL_PATH, map_location=DEVICE)
                )
                self.model.eval()
                debug_log(f"Model loaded from {MODEL_PATH}")
            else:
                debug_log(f"Warning: Model file {MODEL_PATH} not found")

            return True
        except Exception as e:
            debug_log(f"Error initializing model: {str(e)}")
            return False
