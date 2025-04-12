from src.config import (
    DATA_PATH,
    DEVICE,
    HIDDEN_DIM,
    NUM_HEADS,
    NUM_LAYERS
)
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, Linear, HGTConv, GraphNorm, HeteroDictLinear
from torch_geometric.nn import TransformerConv, BatchNorm, to_hetero
from torch_geometric.utils import add_self_loops
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter
from sentence_transformers import SentenceTransformer
import os
import time
import traceback
import warnings
warnings.filterwarnings('ignore')


# Set device
device = DEVICE
print(f"Using device: {device}")

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# Load the JSON data
print("Loading job data...")
try:
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    print(f"Loaded data with {len(data)} job titles")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Load the Sentence Transformer model
print("Loading Sentence Transformer model...")
model_name = "sentence-transformers/msmarco-distilbert-base-v4"
sentence_transformer = SentenceTransformer(model_name)
print(f"Loaded {model_name}")


class HeteroJobRecommendationSystem:
    def __init__(self, data):
        self.data = data

        # Dictionaries for tracking entity mappings and relationships
        # Dictionary to store hotness score of each technology (0-1 scale)
        self.tech_hotness = {}
        # Dictionary to store demand score of each technology (0-1 scale)
        self.tech_demand = {}
        # Dictionary to store demand score of each job title (0-1 scale)
        self.job_demand = {}
        self.job_hot_tech_count = {}  # Dictionary to count hot technologies for each job

        # Create mappings for graph construction
        self.all_job_titles = []
        self.all_skills = []
        self.all_technologies = []

        # TF-IDF related counters
        self.skill_job_count = Counter()  # How many jobs require each skill
        self.tech_job_count = Counter()   # How many jobs require each technology
        self.job_skill_count = {}  # Number of skills per job
        self.job_tech_count = {}   # Number of technologies per job

        # Importance dictionaries
        self.skill_importance_by_job = {}  # Dict to store skill importance for each job
        # Dict to store technology importance for each job
        self.tech_importance_by_job = {}

        # Node and edge mappings
        self.job_skill_tech_map = {}  # Maps jobs to their skills and technologies

        # Processed data
        self.unique_job_titles = None
        self.unique_skills = None
        self.unique_technologies = None
        self.num_jobs = 0
        self.num_skills = 0
        self.num_techs = 0

        # Encoders
        self.job_encoder = None
        self.skill_encoder = None
        self.tech_encoder = None

        # IDF values
        self.skill_idf = None
        self.tech_idf = None

        # Graph data
        self.hetero_data = None
        self.node_types = None
        self.edge_types = None
        self.metadata = None

        # Feature embeddings
        self.job_tensor = None
        self.skill_tensor = None
        self.tech_tensor = None
        self.job_features = None
        self.skill_features = None
        self.tech_features = None

    def extract_metrics(self):
        """Extract demand and hotness metrics from the data"""
        print("Extracting technology hotness and demand metrics...")

        for job in tqdm(self.data):
            job_title = job['title']
            job_hot_techs = 0
            job_total_techs = 0
            tech_demand_scores = []

            for skill_data in job['technology_skills']:
                for tech in skill_data['technologies']:
                    tech_name = tech['name']
                    job_total_techs += 1

                    # Extract hot_tech_percentage and convert to float
                    try:
                        hot_tech_value = float(
                            tech.get('hot_tech_percentage', '0'))
                        # Convert to 0-1 scale
                        self.tech_hotness[tech_name] = hot_tech_value / 100.0
                        # Count as hot technology if hot_tech is "Yes"
                        if tech.get('hot_tech', 'No').lower() == 'yes':
                            job_hot_techs += 1
                    except (ValueError, TypeError):
                        self.tech_hotness[tech_name] = 0.0

                    # Extract demand_percentage and convert to float
                    try:
                        demand_value = float(
                            tech.get('demand_percentage', '0'))
                        # Convert to 0-1 scale
                        self.tech_demand[tech_name] = demand_value / 100.0
                        tech_demand_scores.append(demand_value / 100.0)
                    except (ValueError, TypeError):
                        self.tech_demand[tech_name] = 0.0

            # Calculate job demand as average of its technology demand scores
            if tech_demand_scores:
                self.job_demand[job_title] = sum(
                    tech_demand_scores) / len(tech_demand_scores)
            else:
                self.job_demand[job_title] = 0.0

            # Store the ratio of hot technologies for each job
            if job_total_techs > 0:
                self.job_hot_tech_count[job_title] = job_hot_techs / \
                    job_total_techs
            else:
                self.job_hot_tech_count[job_title] = 0.0

        print(
            f"Extracted hotness scores for {len(self.tech_hotness)} technologies")
        print(
            f"Extracted demand scores for {len(self.tech_demand)} technologies")
        print(
            f"Calculated demand scores for {len(self.job_demand)} job titles")

    def extract_entities(self):
        """Extract jobs, skills, and technologies from the data"""
        print("Extracting jobs, skills, and technologies...")

        for job in tqdm(self.data):
            job_title = job['title']
            self.all_job_titles.append(job_title)

            job_skills = set()  # Use set to avoid counting duplicates within a job
            job_techs = set()

            for skill_data in job['technology_skills']:
                skill_title = skill_data['skill_title']
                self.all_skills.append(skill_title)
                job_skills.add(skill_title)

                for tech in skill_data['technologies']:
                    tech_name = tech['name']
                    self.all_technologies.append(tech_name)
                    job_techs.add(tech_name)

            # Update counters
            for skill in job_skills:
                self.skill_job_count[skill] += 1

            for tech in job_techs:
                self.tech_job_count[tech] += 1

            self.job_skill_count[job_title] = len(job_skills)
            self.job_tech_count[job_title] = len(job_techs)

        # Remove duplicates
        self.unique_job_titles = list(set(self.all_job_titles))
        self.unique_skills = list(set(self.all_skills))
        self.unique_technologies = list(set(self.all_technologies))

        self.num_jobs = len(self.unique_job_titles)
        self.num_skills = len(self.unique_skills)
        self.num_techs = len(self.unique_technologies)

        print(f"Unique job titles: {self.num_jobs}")
        print(f"Unique skills: {self.num_skills}")
        print(f"Unique technologies: {self.num_techs}")

        # Define node and edge types for heterogeneous graph
        self.node_types = ['job', 'skill', 'tech']
        self.edge_types = [
            ('job', 'requires', 'skill'),
            ('skill', 'required_by', 'job'),
            ('skill', 'includes', 'tech'),
            ('tech', 'included_in', 'skill')
        ]
        self.metadata = (self.node_types, self.edge_types)

    def create_encoders(self):
        """Create label encoders for jobs, skills, and technologies"""
        print("Creating label encoders...")

        # Calculate IDF (Inverse Document Frequency) for skills and technologies
        total_jobs = len(self.unique_job_titles)
        self.skill_idf = {skill: np.log(total_jobs / (count + 1))
                          for skill, count in self.skill_job_count.items()}
        self.tech_idf = {tech: np.log(total_jobs / (count + 1))
                         for tech, count in self.tech_job_count.items()}

        # Create label encoders
        self.job_encoder = LabelEncoder()
        self.job_encoder.fit(self.unique_job_titles)

        self.skill_encoder = LabelEncoder()
        self.skill_encoder.fit(self.unique_skills)

        self.tech_encoder = LabelEncoder()
        self.tech_encoder.fit(self.unique_technologies)

    def generate_embeddings(self):
        """Generate embeddings for jobs, skills, and technologies"""
        print("Generating embeddings using Sentence Transformer...")

        # Get embedding dimension
        embedding_dim = sentence_transformer.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {embedding_dim}")

        # Generate embeddings in batches
        print("Computing embeddings for jobs...")
        job_embeddings = sentence_transformer.encode(
            self.unique_job_titles, show_progress_bar=True)

        print("Computing embeddings for skills...")
        skill_embeddings = sentence_transformer.encode(
            self.unique_skills, show_progress_bar=True)

        print("Computing embeddings for technologies...")
        tech_embeddings = sentence_transformer.encode(
            self.unique_technologies, show_progress_bar=True)

        # Convert to torch tensors AND move to device
        self.job_tensor = torch.tensor(
            job_embeddings, dtype=torch.float).to(device)
        self.skill_tensor = torch.tensor(
            skill_embeddings, dtype=torch.float).to(device)
        self.tech_tensor = torch.tensor(
            tech_embeddings, dtype=torch.float).to(device)

        # Create embeddings from tensors
        self.job_features = nn.Embedding.from_pretrained(
            self.job_tensor, freeze=False)
        self.skill_features = nn.Embedding.from_pretrained(
            self.skill_tensor, freeze=False)
        self.tech_features = nn.Embedding.from_pretrained(
            self.tech_tensor, freeze=False)

    def build_graph(self):
        """Build the heterogeneous graph with nodes and edges"""
        print("Building the heterogeneous graph...")

        # Create a heterogeneous graph data structure
        hetero_data = HeteroData()

        # Add node features for each node type
        # For jobs, add base embedding features plus demand and hotness
        job_demand_features = torch.zeros(self.num_jobs, 1, device=device)
        job_hotness_features = torch.zeros(self.num_jobs, 1, device=device)

        # Fill in job demand and hotness values
        for job_title, job_id in zip(self.unique_job_titles, range(self.num_jobs)):
            job_demand_features[job_id, 0] = self.job_demand.get(
                job_title, 0.0)
            job_hotness_features[job_id, 0] = self.job_hot_tech_count.get(
                job_title, 0.0)

        # Concatenate base embeddings with demand and hotness for job nodes
        job_node_features = torch.cat(
            [self.job_tensor, job_demand_features, job_hotness_features], dim=1)

        # Add job node features to graph
        hetero_data['job'].x = job_node_features

        # For skills, add embeddings
        hetero_data['skill'].x = self.skill_tensor

        # For technologies, add embeddings, demand, and hotness
        tech_demand_features = torch.zeros(self.num_techs, 1, device=device)
        tech_hotness_features = torch.zeros(self.num_techs, 1, device=device)

        # Fill in technology demand and hotness values
        for tech_name, tech_id in zip(self.unique_technologies, range(self.num_techs)):
            tech_demand_features[tech_id,
                                 0] = self.tech_demand.get(tech_name, 0.0)
            tech_hotness_features[tech_id,
                                  0] = self.tech_hotness.get(tech_name, 0.0)

        # Concatenate base embeddings with demand and hotness for tech nodes
        tech_node_features = torch.cat(
            [self.tech_tensor, tech_demand_features, tech_hotness_features], dim=1)

        # Add tech node features to graph
        hetero_data['tech'].x = tech_node_features

        # Create the edge indices and edge attributes for each edge type
        job_to_skill_edges = []
        skill_to_job_edges = []
        skill_to_tech_edges = []
        tech_to_skill_edges = []

        # Edge attributes for different relation types
        job_skill_weights = []
        skill_job_weights = []
        skill_tech_weights = []
        tech_skill_weights = []

        # Edge features for heterogeneous edges
        job_skill_features = []
        skill_tech_features = []

        # Track which skills and technologies are associated with each job
        for job in tqdm(self.data):
            job_title = job['title']
            job_id = self.job_encoder.transform([job_title])[0]

            if job_title not in self.job_skill_tech_map:
                self.job_skill_tech_map[job_title] = {
                    'skills': set(), 'technologies': set()}
                self.skill_importance_by_job[job_title] = {}
                self.tech_importance_by_job[job_title] = {}

            # Count skill and tech occurrences within this job
            job_skill_count_local = Counter()
            job_tech_count_local = Counter()

            for skill_data in job['technology_skills']:
                skill_title = skill_data['skill_title']
                skill_id = self.skill_encoder.transform([skill_title])[0]

                # Add job-skill edge
                job_to_skill_edges.append((job_id, skill_id))
                skill_to_job_edges.append((skill_id, job_id))

                # Calculate edge weight using TF-IDF
                skill_idf_value = self.skill_idf.get(skill_title, 1.0)
                job_skill_weights.append(skill_idf_value)
                skill_job_weights.append(skill_idf_value)

                # Add edge feature (TF-IDF) as a tensor
                job_skill_features.append([skill_idf_value])

                self.job_skill_tech_map[job_title]['skills'].add(skill_title)
                job_skill_count_local[skill_title] += 1

                for tech in skill_data['technologies']:
                    tech_name = tech['name']
                    tech_id = self.tech_encoder.transform([tech_name])[0]

                    # Get hotness and demand values for this tech
                    tech_hotness_value = self.tech_hotness.get(tech_name, 0.0)
                    tech_demand_value = self.tech_demand.get(tech_name, 0.0)

                    # Add skill-tech edge
                    skill_to_tech_edges.append((skill_id, tech_id))
                    tech_to_skill_edges.append((tech_id, skill_id))

                    # Calculate edge weight using TF-IDF
                    tech_idf_value = self.tech_idf.get(tech_name, 1.0)
                    skill_tech_weights.append(tech_idf_value)
                    tech_skill_weights.append(tech_idf_value)

                    # Add edge feature as a tensor (TF-IDF, hotness, demand)
                    skill_tech_features.append(
                        [tech_idf_value, tech_hotness_value, tech_demand_value])

                    self.job_skill_tech_map[job_title]['technologies'].add(
                        tech_name)
                    job_tech_count_local[tech_name] += 1

            # Calculate skill importance for this job using TF-IDF
            for skill, count in job_skill_count_local.items():
                # Term frequency
                skill_tf = count / max(len(job_skill_count_local), 1)
                self.skill_importance_by_job[job_title][skill] = skill_tf * \
                    self.skill_idf.get(skill, 1.0)

            # Calculate technology importance for this job using TF-IDF
            for tech, count in job_tech_count_local.items():
                # Term frequency
                tech_tf = count / max(len(job_tech_count_local), 1)
                self.tech_importance_by_job[job_title][tech] = tech_tf * \
                    self.tech_idf.get(tech, 1.0)

        # Convert edge indices to tensors and add to heterogeneous data
        if job_to_skill_edges:
            job_to_skill = torch.tensor(
                job_to_skill_edges, dtype=torch.long).t().contiguous().to(device)
            skill_to_job = torch.tensor(
                skill_to_job_edges, dtype=torch.long).t().contiguous().to(device)

            # Add edges to the graph
            hetero_data[('job', 'requires', 'skill')].edge_index = job_to_skill
            hetero_data[('skill', 'required_by', 'job')
                        ].edge_index = skill_to_job

            # Add edge attributes
            job_skill_attr = torch.tensor(
                job_skill_features, dtype=torch.float).to(device)
            hetero_data[('job', 'requires', 'skill')
                        ].edge_attr = job_skill_attr
            skill_job_attr = torch.tensor(
                [[w] for w in skill_job_weights], dtype=torch.float).to(device)
            hetero_data[('skill', 'required_by', 'job')
                        ].edge_attr = skill_job_attr

        if skill_to_tech_edges:
            skill_to_tech = torch.tensor(
                skill_to_tech_edges, dtype=torch.long).t().contiguous().to(device)
            tech_to_skill = torch.tensor(
                tech_to_skill_edges, dtype=torch.long).t().contiguous().to(device)

            # Add edges to the graph
            hetero_data[('skill', 'includes', 'tech')
                        ].edge_index = skill_to_tech
            hetero_data[('tech', 'included_in', 'skill')
                        ].edge_index = tech_to_skill

            # Add edge attributes
            skill_tech_attr = torch.tensor(
                skill_tech_features, dtype=torch.float).to(device)
            hetero_data[('skill', 'includes', 'tech')
                        ].edge_attr = skill_tech_attr
            tech_skill_attr = torch.tensor(
                [[w] for w in tech_skill_weights], dtype=torch.float).to(device)
            hetero_data[('tech', 'included_in', 'skill')
                        ].edge_attr = tech_skill_attr

        # Create train/val split for job nodes
        num_train = int(0.8 * self.num_jobs)
        train_mask = torch.zeros(self.num_jobs, dtype=torch.bool).to(device)
        val_mask = torch.zeros(self.num_jobs, dtype=torch.bool).to(device)

        # Random split
        perm = torch.randperm(self.num_jobs).to(device)
        train_idx = perm[:num_train]
        val_idx = perm[num_train:]

        train_mask[train_idx] = True
        val_mask[val_idx] = True

        # Add masks and labels to job nodes
        hetero_data['job'].train_mask = train_mask
        hetero_data['job'].val_mask = val_mask
        hetero_data['job'].y = torch.arange(
            self.num_jobs).to(device)  # Self-supervision

        # Create bidirectional connections within the same node type for message passing
        # This helps with information flow in the graph
        self.hetero_data = hetero_data

        print(
            f"Created heterogeneous graph with node types: {self.hetero_data.node_types}")
        print(f"Edge types: {self.hetero_data.edge_types}")
        for edge_type in self.hetero_data.edge_types:
            print(
                f"  {edge_type}: {self.hetero_data[edge_type].edge_index.size(1)} edges")

        return hetero_data

    def process_data(self):
        """Process all data and build the graph"""
        self.extract_metrics()
        self.extract_entities()
        self.create_encoders()
        self.generate_embeddings()
        return self.build_graph()


class HeteroDemandHotnessPredictor(torch.nn.Module):
    """Auxiliary model to predict demand/hotness from node embeddings"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.demand_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.hotness_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        demand = self.demand_predictor(x)
        hotness = self.hotness_predictor(x)
        return demand, hotness


class HGNNJobRecommender(nn.Module):
    """Heterogeneous Graph Neural Network for job recommendation"""

    def __init__(self, hetero_data, hidden_dim=256, num_heads=4, num_layers=2):
        super(HGNNJobRecommender, self).__init__()

        # Get metadata (node types and edge types)
        self.node_types = hetero_data.node_types
        self.edge_types = hetero_data.edge_types
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Get feature dimensions for each node type
        self.node_feature_dims = {
            node_type: hetero_data[node_type].x.size(1)
            for node_type in self.node_types
        }

        # Create node type-specific input projection layers
        self.input_projectors = nn.ModuleDict({
            node_type: nn.Linear(dim, hidden_dim)
            for node_type, dim in self.node_feature_dims.items()
        })

        # Create Heterogeneous Graph Transformer (HGT) layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=hetero_data.metadata(),
                heads=num_heads
            )
            self.convs.append(conv)

        # Batch normalization for each node type
        self.batch_norms = nn.ModuleDict({
            node_type: nn.BatchNorm1d(hidden_dim)
            for node_type in self.node_types
        })

        # Node type-specific output layers
        self.job_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            # num_jobs
            nn.Linear(hidden_dim, hetero_data['job'].y.max().item() + 1),
        )

        # Auxiliary task predictors
        self.auxiliary_predictors = nn.ModuleDict({
            'job': HeteroDemandHotnessPredictor(hidden_dim),
            'tech': HeteroDemandHotnessPredictor(hidden_dim)
        })

        # Activation and regularization
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_dict, edge_indices_dict, edge_attr_dict=None):
        # Project node features to the same hidden dimension
        h_dict = {
            node_type: self.input_projectors[node_type](x)
            for node_type, x in x_dict.items()
        }

        # Apply HGT layers
        for conv in self.convs:
            h_dict_new = conv(h_dict, edge_indices_dict)

            # Apply batch normalization and non-linearity
            for node_type in h_dict_new.keys():
                h_dict_new[node_type] = self.batch_norms[node_type](
                    h_dict_new[node_type])
                h_dict_new[node_type] = self.leaky_relu(h_dict_new[node_type])
                h_dict_new[node_type] = self.dropout(h_dict_new[node_type])

            h_dict = h_dict_new

        # Apply output layers for specific tasks
        job_logits = self.job_classifier(h_dict['job'])

        # Predict demand and hotness for job and technology nodes
        job_demand, job_hotness = self.auxiliary_predictors['job'](
            h_dict['job'])
        tech_demand, tech_hotness = self.auxiliary_predictors['tech'](
            h_dict['tech'])

        return job_logits, {
            'job_demand': job_demand,
            'job_hotness': job_hotness,
            'tech_demand': tech_demand,
            'tech_hotness': tech_hotness,
        }, h_dict
