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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# Check if running on Kaggle
IN_KAGGLE = os.path.exists('/kaggle/input')
print(f"Running in Kaggle environment: {IN_KAGGLE}")

# Paths configuration
if IN_KAGGLE:
    # Kaggle paths
    BASE_PATH = '/kaggle/input'
    DATA_PATH = f'{BASE_PATH}/onet-data-filtered/filtered_IT_data.json'
    OUTPUT_PATH = '/kaggle/working'
else:
    # Local paths
    BASE_PATH = '.'
    DATA_PATH = 'filtered_IT_data.json'
    OUTPUT_PATH = '.'

# Display paths for debugging
print(f"Data path: {DATA_PATH}")
print(f"Output path: {OUTPUT_PATH}")

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


class EnhancedJobRecommendationSystem:
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
        self.graph_data = None
        self.edge_index = None
        self.edge_weights = None
        self.edge_features = None
        self.node_features = None

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
        print("Building the graph...")

        # Create the edge indices (connections between nodes)
        job_to_skill_edges = []
        skill_to_tech_edges = []

        # Create lists for edge attributes/features
        edge_weights = []  # Basic weights based on TF-IDF
        edge_hotness = []  # Hotness values for edges
        edge_demand = []   # Demand values for edges

        # Track which skills and technologies are associated with each job
        for job_idx, job in enumerate(tqdm(self.data)):
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
                job_to_skill_edges.append((job_id, skill_id + self.num_jobs))
                job_to_skill_edges.append(
                    (skill_id + self.num_jobs, job_id))  # Bidirectional

                # Calculate edge weight using TF-IDF
                skill_idf_value = self.skill_idf.get(skill_title, 1.0)
                edge_weights.append(skill_idf_value)
                edge_weights.append(skill_idf_value)  # For bidirectional edge

                # Add placeholders for hotness and demand on skill edges
                # (these will be updated with tech values later)
                edge_hotness.append(0.0)
                edge_hotness.append(0.0)
                edge_demand.append(0.0)
                edge_demand.append(0.0)

                self.job_skill_tech_map[job_title]['skills'].add(skill_title)
                job_skill_count_local[skill_title] += 1

                for tech in skill_data['technologies']:
                    tech_name = tech['name']
                    tech_id = self.tech_encoder.transform([tech_name])[0]

                    # Get hotness and demand values for this tech
                    tech_hotness_value = self.tech_hotness.get(tech_name, 0.0)
                    tech_demand_value = self.tech_demand.get(tech_name, 0.0)

                    # Add skill-tech edge
                    skill_to_tech_edges.append(
                        (skill_id + self.num_jobs, tech_id + self.num_jobs + self.num_skills))
                    skill_to_tech_edges.append(
                        # Bidirectional
                        (tech_id + self.num_jobs + self.num_skills, skill_id + self.num_jobs))

                    # Calculate edge weight using TF-IDF
                    tech_idf_value = self.tech_idf.get(tech_name, 1.0)
                    edge_weights.append(tech_idf_value)
                    # For bidirectional edge
                    edge_weights.append(tech_idf_value)

                    # Add hotness and demand values as edge attributes
                    # Forward edge
                    edge_hotness.append(tech_hotness_value)
                    edge_demand.append(tech_demand_value)
                    # Backward edge
                    edge_hotness.append(tech_hotness_value)
                    edge_demand.append(tech_demand_value)

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

        # Convert lists to tensors AND move to device
        job_to_skill = torch.tensor(
            job_to_skill_edges, dtype=torch.long).to(device).t().contiguous()
        skill_to_tech = torch.tensor(
            skill_to_tech_edges, dtype=torch.long).to(device).t().contiguous()

        # Combine edge indices for the full graph
        self.edge_index = torch.cat([job_to_skill, skill_to_tech], dim=1)

        # Convert edge attributes to tensors and move to device
        self.edge_weights = torch.tensor(
            edge_weights, dtype=torch.float).to(device)

        # Create a combined edge feature tensor (weight, hotness, demand)
        edge_hotness_tensor = torch.tensor(
            edge_hotness, dtype=torch.float).to(device)
        edge_demand_tensor = torch.tensor(
            edge_demand, dtype=torch.float).to(device)

        # Combine all edge features into a single tensor [weight, hotness, demand]
        self.edge_features = torch.stack([
            self.edge_weights,
            edge_hotness_tensor,
            edge_demand_tensor
        ], dim=1)

        print(f"Created graph with {self.edge_index.size(1)} edges")
        print(f"Edge features shape: {self.edge_features.shape}")

    def get_node_features(self):
        """Get node features from pretrained embeddings"""
        # Get initial node embeddings from pretrained embeddings
        job_node_features = self.job_features.weight
        skill_node_features = self.skill_features.weight
        tech_node_features = self.tech_features.weight

        # Create additional node features for jobs, skills, and technologies
        job_demand_features = torch.zeros(self.num_jobs, 1, device=device)
        job_hotness_features = torch.zeros(self.num_jobs, 1, device=device)
        skill_demand_features = torch.zeros(self.num_skills, 1, device=device)
        skill_hotness_features = torch.zeros(self.num_skills, 1, device=device)
        tech_demand_features = torch.zeros(self.num_techs, 1, device=device)
        tech_hotness_features = torch.zeros(self.num_techs, 1, device=device)

        # Fill in the demand and hotness features
        for job_title, job_id in zip(self.unique_job_titles, range(self.num_jobs)):
            job_demand_features[job_id, 0] = self.job_demand.get(
                job_title, 0.0)
            job_hotness_features[job_id, 0] = self.job_hot_tech_count.get(
                job_title, 0.0)

        for tech_name, tech_id in zip(self.unique_technologies, range(self.num_techs)):
            tech_demand_features[tech_id,
                                 0] = self.tech_demand.get(tech_name, 0.0)
            tech_hotness_features[tech_id,
                                  0] = self.tech_hotness.get(tech_name, 0.0)

        # Get embedding dimension
        emb_dim = job_node_features.shape[1]

        # Concatenate base embeddings with demand and hotness for each node type
        job_features_augmented = torch.cat(
            [job_node_features, job_demand_features, job_hotness_features], dim=1)
        skill_features_augmented = torch.cat(
            [skill_node_features, skill_demand_features, skill_hotness_features], dim=1)
        tech_features_augmented = torch.cat(
            [tech_node_features, tech_demand_features, tech_hotness_features], dim=1)

        # Combine all node features
        all_node_features = torch.cat([
            job_features_augmented,
            skill_features_augmented,
            tech_features_augmented
        ], dim=0)

        # Save final node features
        self.node_features = all_node_features

        return all_node_features

    def create_pyg_data(self):
        """Create PyG data object with node features and edge features"""
        # Get node features
        x = self.get_node_features()

        # Create graph data with edge features
        data = Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_features
        )

        # Create labels for job nodes (just their own indices for pretraining)
        labels = torch.arange(self.num_jobs).to(device)

        # Add labels to data
        data.y = labels

        # Create train/val split for better evaluation
        num_train = int(0.8 * self.num_jobs)
        train_mask = torch.zeros(self.num_jobs, dtype=torch.bool).to(device)
        val_mask = torch.zeros(self.num_jobs, dtype=torch.bool).to(device)

        # Random split
        perm = torch.randperm(self.num_jobs).to(device)
        train_idx = perm[:num_train]
        val_idx = perm[num_train:]

        train_mask[train_idx] = True
        val_mask[val_idx] = True

        data.train_mask = train_mask
        data.val_mask = val_mask

        # Save graph data
        self.graph_data = data

        return data

    def process_data(self):
        """Process all data and build the graph"""
        self.extract_metrics()
        self.extract_entities()
        self.create_encoders()
        self.generate_embeddings()
        self.build_graph()
        return self.create_pyg_data()


# Define an improved GNN model with attention to demand and hotness
class EnhancedGNNJobRecommender(nn.Module):
    def __init__(self, num_jobs, num_skills, num_techs, embedding_dim, edge_dim=3, hidden_dim=256):
        super(EnhancedGNNJobRecommender, self).__init__()

        # Adjusted embedding dimension to account for added demand and hotness features
        self.embedding_dim = embedding_dim + 2  # +2 for demand and hotness
        self.edge_dim = edge_dim  # weight, hotness, demand
        self.hidden_dim = hidden_dim
        self.num_jobs = num_jobs
        self.num_skills = num_skills
        self.num_techs = num_techs

        # Edge feature processing
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Graph convolutional layers with edge feature processing
        self.conv1 = TransformerConv(
            self.embedding_dim, hidden_dim, heads=4, edge_dim=edge_dim)
        self.conv2 = TransformerConv(
            hidden_dim * 4, hidden_dim, heads=4, edge_dim=edge_dim)
        self.conv3 = TransformerConv(
            hidden_dim * 4, hidden_dim, heads=4, edge_dim=edge_dim)

        # Batch normalization
        self.batch_norm1 = BatchNorm(hidden_dim * 4)
        self.batch_norm2 = BatchNorm(hidden_dim * 4)
        self.batch_norm3 = BatchNorm(hidden_dim * 4)

        # Output MLP for job prediction
        self.job_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_jobs),
        )

        # Demand and hotness prediction (auxiliary tasks)
        self.demand_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.hotness_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Activation and regularization
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_attr=None):
        # Initial graph convolution with edge features
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.leaky_relu(x)
        x = self.batch_norm1(x)
        x = self.dropout(x)

        # Second graph convolution
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.leaky_relu(x)
        x = self.batch_norm2(x)
        x = self.dropout(x)

        # Third graph convolution
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.leaky_relu(x)
        x = self.batch_norm3(x)
        x = self.dropout(x)

        # Get job node embeddings
        job_embeddings = x[:self.num_jobs]

        # Predict job classes
        job_logits = self.job_classifier(job_embeddings)

        # Predict demand and hotness (auxiliary tasks)
        demand_scores = self.demand_predictor(job_embeddings)
        hotness_scores = self.hotness_predictor(job_embeddings)

        return job_logits, demand_scores, hotness_scores


# Training functions
def train(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    job_logits, demand_scores, hotness_scores = model(
        data.x, data.edge_index, data.edge_attr)

    # Calculate job classification loss
    job_loss = criterion['classification'](
        job_logits[data.train_mask], data.y[data.train_mask])

    # Calculate auxiliary losses (demand and hotness prediction)
    # Extract ground truth values from the node features (embedding_dim is the base dimension)
    embedding_dim = data.x.size(1) - 2
    demand_targets = data.x[:model.num_jobs, embedding_dim].unsqueeze(1)
    hotness_targets = data.x[:model.num_jobs, embedding_dim + 1].unsqueeze(1)

    # MSE loss for demand and hotness prediction
    demand_loss = criterion['regression'](
        demand_scores[data.train_mask], demand_targets[data.train_mask])
    hotness_loss = criterion['regression'](
        hotness_scores[data.train_mask], hotness_targets[data.train_mask])

    # Combine losses with weighting
    loss = job_loss + 0.3 * demand_loss + 0.3 * hotness_loss

    # Backward pass and optimize
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        'total_loss': loss.item(),
        'job_loss': job_loss.item(),
        'demand_loss': demand_loss.item(),
        'hotness_loss': hotness_loss.item()
    }


def evaluate(model, criterion, data, mask):
    model.eval()
    with torch.no_grad():
        # Forward pass
        job_logits, demand_scores, hotness_scores = model(
            data.x, data.edge_index, data.edge_attr)

        # Calculate classification metrics
        pred = job_logits[mask].argmax(dim=1)
        correct = (pred == data.y[mask]).sum().item()
        total = mask.sum().item()
        accuracy = correct / total

        # Calculate job classification loss
        job_loss = criterion['classification'](job_logits[mask], data.y[mask])

        # Calculate auxiliary losses
        embedding_dim = data.x.size(1) - 2
        demand_targets = data.x[:model.num_jobs, embedding_dim].unsqueeze(1)
        hotness_targets = data.x[:model.num_jobs,
                                 embedding_dim + 1].unsqueeze(1)

        demand_loss = criterion['regression'](
            demand_scores[mask], demand_targets[mask])
        hotness_loss = criterion['regression'](
            hotness_scores[mask], hotness_targets[mask])

        # Combine losses
        loss = job_loss + 0.3 * demand_loss + 0.3 * hotness_loss

    return {
        'accuracy': accuracy,
        'total_loss': loss.item(),
        'job_loss': job_loss.item(),
        'demand_loss': demand_loss.item(),
        'hotness_loss': hotness_loss.item()
    }


# Job recommendation function that uses the GNN model more heavily
def predict_job_titles(model, data, system, user_input_skills, top_k=5):
    """
    Predict job titles using the GNN model with minimal rule-based components

    Parameters:
    model: Trained GNN model
    data: PyG data object
    system: EnhancedJobRecommendationSystem object
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
                # Fix incorrect variable name - use skill_name instead of tech_name
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

    # Get model's job embeddings for similarity calculation
    with torch.no_grad():
        job_logits, demand_scores, hotness_scores = model(
            data.x, data.edge_index, data.edge_attr)
        # Get softmax probabilities
        job_probs = torch.softmax(job_logits, dim=1)
        # Get the job embeddings from the last layer
        job_embeddings = model.conv3(model.conv2(model.conv1(
            data.x, data.edge_index, data.edge_attr),
            data.edge_index, data.edge_attr),
            data.edge_index, data.edge_attr)[:system.num_jobs]

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
        tech_count = 0
        skill_count = 0
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

        # Get GNN-predicted demand and hotness scores for this job
        predicted_demand = demand_scores[job_id].item()
        predicted_hotness = hotness_scores[job_id].item()

        # Get job's self-prediction probability and use as confidence
        job_confidence = job_probs[job_id, job_id].item()

        # Calculate match indicator (both skill and tech matches)
        match_indicator = min(skill_count + tech_count, 10) / 10

        # Create a combined score - much more weight on GNN predictions
        # Very minimal rule-based component
        combined_score = (
            0.65 * job_confidence +               # Major weight on model confidence
            0.15 * match_indicator +              # Small weight on having direct matches
            0.10 * tech_coverage +                # Small weight on tech coverage
            0.05 * skill_coverage +               # Small weight on skill coverage
            0.03 * predicted_demand +             # Very small weight on predicted demand
            0.02 * predicted_hotness              # Very small weight on predicted hotness
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


# Example user inputs for testing (same as in kaggle_job_inference.py)
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
    },

    # DevOps Engineer example input
    {
        "name": "Computer Network Support Specialists",
        "skills": [
            ("SolarWinds", "technology_name", 1.00),
            ("Wireshark", "technology_name", 1.00),
            ("Firewall software (Firewall)", "technology_name", 0.96),
            ("Network monitoring software (Net Monitor SW)", "skill_title", 0.87),
            ("Cisco Systems CiscoWorks (CiscoWorks) LAN Management Solution (Cisco LMS)",
             "technology_name", 0.74),
            ("Switch software (Switch SW)", "skill_title", 0.69),
            ("router software (Router SW)", "skill_title", 0.66),
            ("design software", "skill_title", 0.57),
            ("Device drivers software", "skill_title", 0.51)
        ]
    },

    # UI/UX Designer example input
    {
        "name": "UI/UX Designer",
        "skills": [
            ("Adobe XD (XD)", "technology_name", 1.00),
            ("Google Angular (Angular)", "technology_name", 1.00),
            ("Cascading style sheets (CSS) CSS", "technology_name", 1.00),
            ("JavaScript (JS) Object Notation JSON (JSON)", "technology_name", 1.00),
            ("Hypertext markup language (HTML)", "technology_name", 1.00),
            ("Figma", "technology_name", 1.00),
            ("Web application software (Web App)", "technology_name", 0.87),
            ("User interface design software (UI Design)", "technology_name", 0.76),
            ("Web framework software (Web FW)", "technology_name", 0.70),
            ("Vue.js (Vue)", "technology_name", 0.66),
            ("Microsoft Dynamics (Dynamics)", "technology_name", 0.64),
            ("Software development tools (Dev Tools)", "technology_name", 0.63),
            ("Microsoft FrontPage (FrontPage)", "technology_name", 0.62),
            ("design software", "skill_title", 0.62),
            ("Interactive voice response software (IVR SW)", "skill_title", 0.60),
            ("Development environment software (Dev Env SW)", "skill_title", 0.58),
            ("Web platform development software (Web Dev SW)", "skill_title", 0.55)
        ]
    }
]


def main():
    """Main execution function"""
    start_time = time.time()
    print(f"Starting enhanced GNN-based job recommendation system...")

    try:
        # Initialize the job recommendation system
        system = EnhancedJobRecommendationSystem(data)

        # Process data and build graph
        graph_data = system.process_data()

        # Set up model
        emb_dim = system.job_tensor.shape[1]
        model = EnhancedGNNJobRecommender(
            system.num_jobs,
            system.num_skills,
            system.num_techs,
            emb_dim,
            edge_dim=3,
            hidden_dim=256
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
        print(f"\nTraining the enhanced GNN model for {num_epochs} epochs...")
        try:
            for epoch in range(num_epochs):
                # Train
                train_results = train(model, optimizer, criterion, graph_data)

                # Evaluate on train set
                train_metrics = evaluate(
                    model, criterion, graph_data, graph_data.train_mask)

                # Evaluate on validation set
                val_metrics = evaluate(
                    model, criterion, graph_data, graph_data.val_mask)

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
        model_path = os.path.join(OUTPUT_PATH, "enhanced_gnn_job_model.pt")
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
        plt.savefig(os.path.join(OUTPUT_PATH, "training_history.png"))

        # Test with example inputs
        print("\nTesting with example inputs...")
        for example in EXAMPLE_INPUTS:
            print(f"\nProcessing example: {example['name']}")

            # Get job recommendations
            top_jobs, job_details = predict_job_titles(
                model, graph_data, system, example['skills'], top_k=5)

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
                OUTPUT_PATH, f"{example['name'].replace(' ', '_')}_recommendations.png")
            visualize_job_recommendations(
                top_jobs,
                job_details,
                title=f"Job Recommendations for {example['name']}",
                save_path=vis_path,
                show_plot=False
            )

        # Run a quick custom test as requested
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
        custom_top_jobs, custom_job_details = predict_job_titles(
            model, graph_data, system, quick_test_input, top_k=38)

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
            OUTPUT_PATH, "custom_skills_recommendations.png")
        visualize_job_recommendations(
            custom_top_jobs,
            custom_job_details,
            title="Job Recommendations for Custom Skills",
            save_path=custom_vis_path,
            show_plot=False
        )

        print(f"\nCustom skills visualization saved to: {custom_vis_path}")

        # Final timing
        end_time = time.time()
        print(f"\nTotal execution time: {(end_time - start_time):.1f} seconds")
        print("Enhanced GNN-based job recommendation completed successfully!")

    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
