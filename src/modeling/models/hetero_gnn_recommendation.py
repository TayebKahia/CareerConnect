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


# Training functions for heterogeneous GNN
def train_hetero_gnn(model, optimizer, criterion, hetero_data):
    model.train()
    optimizer.zero_grad()

    # Prepare inputs for the model
    x_dict = {
        node_type: hetero_data[node_type].x for node_type in hetero_data.node_types}
    edge_index_dict = {
        edge_type: hetero_data[edge_type].edge_index for edge_type in hetero_data.edge_types}
    edge_attr_dict = {edge_type: hetero_data[edge_type].edge_attr for edge_type in hetero_data.edge_types
                      if 'edge_attr' in hetero_data[edge_type]}

    # Forward pass
    job_logits, aux_outputs, _ = model(x_dict, edge_index_dict, edge_attr_dict)

    # Calculate job classification loss
    job_loss = criterion['classification'](
        job_logits[hetero_data['job'].train_mask],
        hetero_data['job'].y[hetero_data['job'].train_mask]
    )

    # Get ground truth values from node features for auxiliary tasks
    # For job nodes
    job_embedding_dim = hetero_data['job'].x.size(
        1) - 2  # Base dim without demand & hotness
    job_demand_targets = hetero_data['job'].x[:,
                                              job_embedding_dim].unsqueeze(1)
    job_hotness_targets = hetero_data['job'].x[:,
                                               job_embedding_dim + 1].unsqueeze(1)

    # For technology nodes
    tech_embedding_dim = hetero_data['tech'].x.size(
        1) - 2  # Base dim without demand & hotness
    tech_demand_targets = hetero_data['tech'].x[:,
                                                tech_embedding_dim].unsqueeze(1)
    tech_hotness_targets = hetero_data['tech'].x[:,
                                                 tech_embedding_dim + 1].unsqueeze(1)

    # Calculate auxiliary losses
    job_demand_loss = criterion['regression'](
        aux_outputs['job_demand'][hetero_data['job'].train_mask],
        job_demand_targets[hetero_data['job'].train_mask]
    )
    job_hotness_loss = criterion['regression'](
        aux_outputs['job_hotness'][hetero_data['job'].train_mask],
        job_hotness_targets[hetero_data['job'].train_mask]
    )
    tech_demand_loss = criterion['regression'](
        aux_outputs['tech_demand'], tech_demand_targets)
    tech_hotness_loss = criterion['regression'](
        aux_outputs['tech_hotness'], tech_hotness_targets)

    # Combine losses with weighting
    loss = (
        job_loss +
        0.2 * job_demand_loss +
        0.2 * job_hotness_loss +
        0.1 * tech_demand_loss +
        0.1 * tech_hotness_loss
    )

    # Backward pass and optimize
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        'total_loss': loss.item(),
        'job_loss': job_loss.item(),
        'job_demand_loss': job_demand_loss.item(),
        'job_hotness_loss': job_hotness_loss.item(),
        'tech_demand_loss': tech_demand_loss.item(),
        'tech_hotness_loss': tech_hotness_loss.item()
    }


def evaluate_hetero_gnn(model, criterion, hetero_data, mask_type='val_mask'):
    model.eval()

    with torch.no_grad():
        # Prepare inputs
        x_dict = {
            node_type: hetero_data[node_type].x for node_type in hetero_data.node_types}
        edge_index_dict = {
            edge_type: hetero_data[edge_type].edge_index for edge_type in hetero_data.edge_types}
        edge_attr_dict = {edge_type: hetero_data[edge_type].edge_attr for edge_type in hetero_data.edge_types
                          if 'edge_attr' in hetero_data[edge_type]}

        # Forward pass
        job_logits, aux_outputs, _ = model(
            x_dict, edge_index_dict, edge_attr_dict)

        # Get mask from hetero_data
        mask = hetero_data['job'][mask_type]

        # Calculate accuracy
        pred = job_logits[mask].argmax(dim=1)
        correct = (pred == hetero_data['job'].y[mask]).sum().item()
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0.0

        # Calculate job classification loss
        job_loss = criterion['classification'](
            job_logits[mask], hetero_data['job'].y[mask])

        # Get ground truth values for auxiliary tasks
        job_embedding_dim = hetero_data['job'].x.size(1) - 2
        job_demand_targets = hetero_data['job'].x[:,
                                                  job_embedding_dim].unsqueeze(1)
        job_hotness_targets = hetero_data['job'].x[:,
                                                   job_embedding_dim + 1].unsqueeze(1)

        tech_embedding_dim = hetero_data['tech'].x.size(1) - 2
        tech_demand_targets = hetero_data['tech'].x[:,
                                                    tech_embedding_dim].unsqueeze(1)
        tech_hotness_targets = hetero_data['tech'].x[:,
                                                     tech_embedding_dim + 1].unsqueeze(1)

        # Calculate auxiliary losses
        job_demand_loss = criterion['regression'](
            aux_outputs['job_demand'][mask], job_demand_targets[mask])
        job_hotness_loss = criterion['regression'](
            aux_outputs['job_hotness'][mask], job_hotness_targets[mask])
        tech_demand_loss = criterion['regression'](
            aux_outputs['tech_demand'], tech_demand_targets)
        tech_hotness_loss = criterion['regression'](
            aux_outputs['tech_hotness'], tech_hotness_targets)

        # Combine losses
        loss = (
            job_loss +
            0.2 * job_demand_loss +
            0.2 * job_hotness_loss +
            0.1 * tech_demand_loss +
            0.1 * tech_hotness_loss
        )

    return {
        'accuracy': accuracy,
        'total_loss': loss.item(),
        'job_loss': job_loss.item(),
        'job_demand_loss': job_demand_loss.item(),
        'job_hotness_loss': job_hotness_loss.item(),
        'tech_demand_loss': tech_demand_loss.item(),
        'tech_hotness_loss': tech_hotness_loss.item()
    }


def predict_job_titles_hetero(model, hetero_data, system, user_input_skills, top_k=5):
    """
    Predict job titles using the heterogeneous GNN model

    Parameters:
    model: Trained heterogeneous GNN model
    hetero_data: PyG Hetero data object
    system: HeteroJobRecommendationSystem object
    user_input_skills: List of tuples (skill_name, skill_type, similarity_score)
    top_k: Number of top job titles to recommend

    Returns:
    List of job title recommendations with confidence scores and explanation details
    """
    model.eval()

    # Map skills and technologies to their IDs
    user_skills = []
    user_techs = []

    for skill_info in user_input_skills:
        skill_name, skill_type, similarity = skill_info

        if skill_type == "skill_title":
            try:
                skill_id = system.skill_encoder.transform([skill_name])[0]
                idf_value = system.skill_idf.get(skill_name, 1.0)
                user_skills.append(
                    (skill_name, skill_id, similarity, idf_value))
            except:
                print(
                    f"Warning: Skill '{skill_name}' not found in training data")
        elif skill_type == "technology_name":
            try:
                tech_id = system.tech_encoder.transform([skill_name])[0]
                idf_value = system.tech_idf.get(skill_name, 1.0)
                hotness_value = system.tech_hotness.get(skill_name, 0.0)
                demand_value = system.tech_demand.get(skill_name, 0.0)
                user_techs.append(
                    (skill_name, tech_id, similarity, idf_value, hotness_value, demand_value))
            except:
                print(
                    f"Warning: Technology '{skill_name}' not found in training data")

    # Create job scores dictionary
    job_scores = {}
    job_match_details = {}

    # Get model's embeddings for similarity calculation
    with torch.no_grad():
        # Prepare inputs
        x_dict = {
            node_type: hetero_data[node_type].x for node_type in hetero_data.node_types}
        edge_index_dict = {
            edge_type: hetero_data[edge_type].edge_index for edge_type in hetero_data.edge_types}
        edge_attr_dict = {edge_type: hetero_data[edge_type].edge_attr for edge_type in hetero_data.edge_types
                          if 'edge_attr' in hetero_data[edge_type]}

        # Forward pass to get node embeddings
        job_logits, aux_outputs, node_embeddings = model(
            x_dict, edge_index_dict, edge_attr_dict)

        # Get job probabilities from logits
        job_probs = torch.softmax(job_logits, dim=1)

        # Get job and tech embeddings from the model
        job_embeddings = node_embeddings['job']
        skill_embeddings = node_embeddings['skill']
        tech_embeddings = node_embeddings['tech']

    # Calculate scores for each job
    for job_title in system.unique_job_titles:
        job_id = system.job_encoder.transform([job_title])[0]

        if job_title not in system.job_skill_tech_map:
            continue

        job_skills = system.job_skill_tech_map[job_title]['skills']
        job_techs = system.job_skill_tech_map[job_title]['technologies']

        # Gather matched skills and technologies for explanation
        skill_matches = []
        tech_matches = []
        skill_count = 0
        tech_count = 0
        matched_techs_with_metrics = []

        # Calculate job embedding similarity with user skills
        for skill_name, skill_id, similarity, idf_value in user_skills:
            if skill_name in job_skills:
                skill_importance = system.skill_importance_by_job.get(
                    job_title, {}).get(skill_name, idf_value)
                skill_matches.append(
                    (skill_name, similarity * skill_importance))
                skill_count += 1

        # Calculate job embedding similarity with user techs
        for tech_name, tech_id, similarity, idf_value, hotness_value, demand_value in user_techs:
            if tech_name in job_techs:
                tech_importance = system.tech_importance_by_job.get(
                    job_title, {}).get(tech_name, idf_value)
                tech_matches.append((tech_name, similarity * tech_importance))
                matched_techs_with_metrics.append(
                    (tech_name, hotness_value, demand_value))
                tech_count += 1

        # Calculate coverage ratios
        total_job_skills = len(job_skills)
        total_job_techs = len(job_techs)

        skill_coverage = skill_count / max(total_job_skills, 1)
        tech_coverage = tech_count / max(total_job_techs, 1)

        # Get demand and hotness values from auxiliary outputs
        predicted_demand = aux_outputs['job_demand'][job_id].item()
        predicted_hotness = aux_outputs['job_hotness'][job_id].item()

        # Get job's self-prediction probability
        job_confidence = job_probs[job_id, job_id].item()

        # Calculate match indicator (both skill and tech matches)
        match_indicator = min(skill_count + tech_count, 10) / 10

        # Create a combined score with high weight on model predictions
        combined_score = (
            0.65 * job_confidence +               # Major weight on model confidence
            0.15 * match_indicator +              # Weight on having direct matches
            0.10 * tech_coverage +                # Weight on tech coverage
            0.05 * skill_coverage +               # Weight on skill coverage
            0.03 * predicted_demand +             # Small weight on predicted demand
            0.02 * predicted_hotness              # Small weight on predicted hotness
        )

        # Store the final score
        job_scores[job_title] = combined_score

        # Calculate display metrics for explanation
        display_hot_tech_percentage = sum([tech[1] for tech in matched_techs_with_metrics]) / max(
            len(matched_techs_with_metrics), 1) if matched_techs_with_metrics else 0.0
        display_demand_percentage = sum([tech[2] for tech in matched_techs_with_metrics]) / max(
            len(matched_techs_with_metrics), 1) if matched_techs_with_metrics else 0.0

        # Store match details for explanation
        job_match_details[job_title] = {
            'skill_matches': skill_matches,
            'tech_matches': tech_matches,
            'skill_coverage': skill_coverage,
            'tech_coverage': tech_coverage,
            'hot_tech_percentage': display_hot_tech_percentage,
            'demand_percentage': display_demand_percentage,
            'matched_techs_with_metrics': matched_techs_with_metrics,
            'gnn_confidence': job_confidence,
            'predicted_demand': predicted_demand,
            'predicted_hotness': predicted_hotness
        }

    # Sort job titles by score
    sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)

    # Get top k job titles
    top_jobs = sorted_jobs[:top_k]

    # Normalize scores to get differentiated confidence percentages (70-100%)
    if top_jobs:
        scores = np.array([score for _, score in top_jobs])
        # Apply sigmoid normalization
        normalized_scores = 1 / \
            (1 + np.exp(-(scores - scores.mean()) / (scores.std() + 1e-6)))
        # Scale to 70-100 range
        scaled_scores = 70 + 30 * (normalized_scores - normalized_scores.min()) / (
            normalized_scores.max() - normalized_scores.min() + 1e-6)

        top_jobs = [(job, score)
                    for (job, _), score in zip(top_jobs, scaled_scores)]

    return top_jobs, job_match_details


# Visualization function
def visualize_job_recommendations(top_jobs, job_match_details, title="Job Recommendations",
                                  save_path=None, show_plot=True):
    """Visualize job recommendations with detailed metrics"""
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1. Bar chart for job confidence scores
    plt.subplot(2, 2, 1)
    jobs = [job for job, _ in top_jobs]
    scores = [score for _, score in top_jobs]

    # Create horizontal bar chart
    y_pos = np.arange(len(jobs))
    plt.barh(y_pos, scores, align='center', color='skyblue')
    plt.yticks(y_pos, jobs)
    plt.xlabel('Confidence Score (%)')
    plt.title('Job Match Confidence')

    # Add values to bars
    for i, v in enumerate(scores):
        plt.text(v + 1, i, f"{v:.1f}%", color='blue', va='center')

    # 2. Technology coverage metrics
    plt.subplot(2, 2, 2)
    tech_coverage = [job_match_details[job]
                     ['tech_coverage'] * 100 for job, _ in top_jobs]
    skill_coverage = [job_match_details[job]
                      ['skill_coverage'] * 100 for job, _ in top_jobs]

    x = np.arange(len(jobs))
    width = 0.35

    plt.bar(x - width/2, tech_coverage, width, label='Technology Coverage')
    plt.bar(x + width/2, skill_coverage, width, label='Skill Coverage')

    plt.xlabel('Job Title')
    plt.ylabel('Coverage (%)')
    plt.title('Skills & Technologies Coverage')
    plt.xticks(x, jobs, rotation=45, ha='right')
    plt.legend()

    # 3. Technology demand and hotness visualization
    plt.subplot(2, 2, 3)
    model_confidence = [job_match_details[job]
                        ['gnn_confidence'] * 100 for job, _ in top_jobs]
    predicted_demand = [job_match_details[job]
                        ['predicted_demand'] * 100 for job, _ in top_jobs]
    predicted_hotness = [job_match_details[job]
                         ['predicted_hotness'] * 100 for job, _ in top_jobs]

    # Create a stacked bar
    plt.bar(x, model_confidence, width, label='GNN Confidence')
    plt.bar(x, predicted_demand, width,
            bottom=model_confidence, label='Predicted Demand')
    plt.bar(x, predicted_hotness, width, bottom=[m+d for m, d in zip(model_confidence, predicted_demand)],
            label='Predicted Hotness')

    plt.xlabel('Job Title')
    plt.ylabel('Score (%)')
    plt.title('GNN Model Predictions')
    plt.xticks(x, jobs, rotation=45, ha='right')
    plt.legend()

    # 4. Number of matched skills and technologies
    plt.subplot(2, 2, 4)
    tech_matches = [len(job_match_details[job]['tech_matches'])
                    for job, _ in top_jobs]
    skill_matches = [len(job_match_details[job]['skill_matches'])
                     for job, _ in top_jobs]

    plt.bar(x - width/2, tech_matches, width, label='Matched Technologies')
    plt.bar(x + width/2, skill_matches, width, label='Matched Skills')

    plt.xlabel('Job Title')
    plt.ylabel('Count')
    plt.title('Matched Skills & Technologies')
    plt.xticks(x, jobs, rotation=45, ha='right')
    plt.legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save if path is provided
    if save_path:
        # Create directory if it doesn't exist
        import os
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Replace spaces and slashes in filename for safety
        safe_path = save_path.replace(' ', '_')

        plt.savefig(safe_path)
        print(f"Visualization saved to {safe_path}")

    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig


# Example user inputs for testing
EXAMPLE_INPUTS = [
    # Frontend Web Developer example input
    {
        "name": "Web Developer",
        "skills": [
            ("MongoDB", "technology_name", 1.00),
            ("React", "technology_name", 1.00),
            ("RESTful (REST API)", "technology_name", 0.98),
            ("Web application software (Web App)", "technology_name", 0.84),
            ("Database management systems (DBMS)", "technology_name", 0.66),
            ("Oracle Developer (SQL Dev)", "technology_name", 0.65),
            ("Amazon Web Services CloudFormation (CloudFormation)",
             "technology_name", 0.63),
            ("Oracle software (Oracle Cloud)", "technology_name", 0.61),
            ("Functional testing software (Func Test SW)", "technology_name", 0.60),
            ("Configuration management software (CM SW)", "skill_title", 0.54),
            ("Workday software (Workday)", "technology_name", 0.54),
            ("Google Android (Android)", "technology_name", 0.54)
        ]
    },

    # Data Scientist example input
    {
        "name": "Data Scientist",
        "skills": [
            ("Python", "technology_name", 1.00),
            ("R", "technology_name", 0.90),
            ("SQL", "technology_name", 0.85),
            ("Machine learning", "skill_title", 0.95),
            ("Data analysis", "skill_title", 1.00),
            ("TensorFlow", "technology_name", 0.80),
            ("PyTorch", "technology_name", 0.75),
            ("Scikit-learn", "technology_name", 0.90),
            ("NumPy", "technology_name", 0.90),
            ("Pandas", "technology_name", 0.90),
            ("Data visualization", "skill_title", 0.80),
            ("Statistics", "skill_title", 0.85)
        ]
    }
]


def main():
    """Main execution function"""
    start_time = time.time()
    print(f"Starting heterogeneous GNN-based job recommendation system...")

    try:
        # Initialize the heterogeneous job recommendation system
        system = HeteroJobRecommendationSystem(data)

        # Process data and build heterogeneous graph
        hetero_data = system.process_data()

        # Set up model
        hidden_dim = 256
        num_heads = 4
        model = HGNNJobRecommender(
            hetero_data,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=2
        ).to(device)

        # Set up optimizer and loss functions
        optimizer = optim.AdamW(
            model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = {
            'classification': nn.CrossEntropyLoss().to(device),
            'regression': nn.MSELoss().to(device)
        }
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        # Training parameters
        num_epochs = 300 if torch.cuda.is_available() else 50  # Fewer epochs for CPU
        patience = 20
        best_val_acc = 0
        patience_counter = 0
        best_model_state = None

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

        # Training loop
        print(
            f"\nTraining the heterogeneous GNN model for {num_epochs} epochs...")
        try:
            for epoch in range(num_epochs):
                # Train
                train_results = train_hetero_gnn(
                    model, optimizer, criterion, hetero_data)

                # Evaluate on train set
                train_metrics = evaluate_hetero_gnn(
                    model, criterion, hetero_data, mask_type='train_mask')

                # Evaluate on validation set
                val_metrics = evaluate_hetero_gnn(
                    model, criterion, hetero_data, mask_type='val_mask')

                # Update learning rate
                scheduler.step(val_metrics['total_loss'])

                # Record history
                history['train_loss'].append(train_results['total_loss'])
                history['train_acc'].append(train_metrics['accuracy'])
                history['val_loss'].append(val_metrics['total_loss'])
                history['val_acc'].append(val_metrics['accuracy'])
                history['lr'].append(optimizer.param_groups[0]['lr'])

                # Print progress every 10 epochs
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/{num_epochs}:")
                    print(
                        f"  Train Loss: {train_results['total_loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
                    print(
                        f"  Val Loss: {val_metrics['total_loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
                    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

                # Check for improvement
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    patience_counter = 0
                    best_model_state = {k: v.cpu()
                                        for k, v in model.state_dict().items()}
                    print(
                        f"  New best validation accuracy: {best_val_acc:.4f}")
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()

        # Load best model
        if best_model_state:
            # First move to CPU to avoid GPU memory issues
            model = model.cpu()
            model.load_state_dict(best_model_state)
            model = model.to(device)
            print(
                f"Loaded best model with validation accuracy: {best_val_acc:.4f}")

        # Save the model
        model_path = os.path.join(OUTPUT_PATH, "hetero_gnn_job_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

        # Plot training history
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.yscale('log')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, "hetero_training_history.png"))

        # Test with example inputs
        print("\nTesting with example inputs...")
        for example in EXAMPLE_INPUTS:
            print(f"\nProcessing example: {example['name']}")

            # Get job recommendations
            top_jobs, job_details = predict_job_titles_hetero(
                model, hetero_data, system, example['skills'], top_k=5)

            # Display results
            print(f"Top job recommendations for {example['name']}:")
            for i, (job_title, confidence) in enumerate(top_jobs, 1):
                details = job_details[job_title]
                print(f"{i}. {job_title}: {confidence:.2f}% match")
                print(
                    f"   - GNN confidence: {details['gnn_confidence']*100:.2f}%")
                print(
                    f"   - Matched {len(details['skill_matches'])} skills and {len(details['tech_matches'])} technologies")

                if details['tech_matches']:
                    top_techs = sorted(
                        details['tech_matches'], key=lambda x: x[1], reverse=True)[:3]
                    print(
                        f"   - Top matching technologies: {', '.join([t[0] for t in top_techs])}")

            # Create visualization
            vis_path = os.path.join(
                OUTPUT_PATH, f"hetero_{example['name'].replace(' ', '_')}_recommendations.png")
            visualize_job_recommendations(
                top_jobs,
                job_details,
                title=f"Hetero GNN Job Recommendations for {example['name']}",
                save_path=vis_path,
                show_plot=False
            )

        # Run a quick custom test
        print("\n" + "="*50)
        print("RUNNING QUICK TEST WITH CUSTOM SKILLS")
        print("="*50)

        # Define custom test input
        quick_test_input = [
            ("Microsoft Azure software (Azure)", "technology_name", 1.00),
            ("NoSQL", "technology_name", 1.00),
            ("Apache", "technology_name", 1.00),
            ("Python", "technology_name", 1.00),
            ("Apache Hadoop (Hadoop)", "technology_name", 1.00),
            ("TensorFlow (TF)", "technology_name", 1.00),
            ("Amazon Web Services SageMaker (SageMaker)", "technology_name", 1.00),
            ("MongoDB", "technology_name", 1.00),
            ("PyTorch", "technology_name", 1.00),
            ("Kubernetes (K8s)", "technology_name", 1.00),
            ("MySQL", "technology_name", 1.00),
            ("Amazon Web Services software (AWS)", "technology_name", 1.00),
            ("Docker", "technology_name", 1.00),
            ("PostgreSQL (Postgres)", "technology_name", 1.00),
            ("IBM Power Systems software (Power)", "technology_name", 1.00),
            ("Tableau", "technology_name", 1.00),
            ("Structured query language (SQL)", "technology_name", 1.00),
            ("R", "technology_name", 1.00),
            ("Git", "technology_name", 1.00),
            ("Microsoft", "technology_name", 1.00),
            ("Version control software (Version Ctrl)", "technology_name", 0.90),
            ("Data visualization software (Data Viz SW)", "technology_name", 0.80)
        ]

        print("\nProcessing custom skills input...")
        custom_top_jobs, custom_job_details = predict_job_titles_hetero(
            model, hetero_data, system, quick_test_input, top_k=5)

        print("\nTop job recommendations for custom skills input:")
        for i, (job_title, confidence) in enumerate(custom_top_jobs, 1):
            details = custom_job_details[job_title]
            print(f"{i}. {job_title}: {confidence:.2f}% match")
            print(f"   - GNN confidence: {details['gnn_confidence']*100:.2f}%")
            print(
                f"   - Matched {len(details['skill_matches'])} skills and {len(details['tech_matches'])} technologies")

            if details['tech_matches']:
                top_techs = sorted(
                    details['tech_matches'], key=lambda x: x[1], reverse=True)[:3]
                print(
                    f"   - Top matching technologies: {', '.join([t[0] for t in top_techs])}")

        # Create visualization for custom test
        custom_vis_path = os.path.join(
            OUTPUT_PATH, "hetero_custom_skills_recommendations.png")
        visualize_job_recommendations(
            custom_top_jobs,
            custom_job_details,
            title="Hetero GNN Job Recommendations for Custom Skills",
            save_path=custom_vis_path,
            show_plot=False
        )

        print(f"\nCustom skills visualization saved to: {custom_vis_path}")

        # Compare to homogeneous model if available
        print("\nHeterogeneous vs. Homogeneous model benefits:")
        print("- Better representation of job-skill-technology relationships")
        print("- Different node types are treated with specialized processing")
        print("- Message passing respects the semantics of different relationships")
        print("- Multi-headed attention captures different relation patterns")
        print("- More explicit modeling of the natural structure of the domain")

        # Final timing
        end_time = time.time()
        print(f"\nTotal execution time: {(end_time - start_time):.1f} seconds")
        print("Heterogeneous GNN-based job recommendation completed successfully!")

    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
