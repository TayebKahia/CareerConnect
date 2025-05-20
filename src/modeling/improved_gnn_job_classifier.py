# This file should be moved to src/modeling/ as it is a modeling-related script.

# Import required modules
import networkx as nx
from fuzzywuzzy import process
from torch_geometric.utils import dense_to_sparse, add_self_loops
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
import time
import os
import re
import json
import numpy as np
import torch
import sys
sys.path.append('.')

# Add the new imports for the improved embeddings
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: SentenceTransformer not available. Install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# Set device and reproducibility
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
np.random.seed(42)
print(f"Using device: {device}")

# Define project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# New function to process O*NET concepts with sentence transformers


def process_onet_concepts(data_path):
    """
    Process O*NET concepts to extract skill titles and technology names
    and generate embeddings using sentence transformers.

    Args:
        data_path (str): Path to the JSON data file

    Returns:
        tuple: processed concepts, main embeddings, abbreviation embeddings
    """
    print(f"Processing O*NET concepts from {data_path}...")

    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        onet_data = json.load(f)

    # Extract skill titles and technology names
    onet_skill_titles = set()
    onet_tech_names = set()

    # Handle both list and dictionary formats
    if isinstance(onet_data, list):
        for job in onet_data:
            # Extract technologies from technology_skills
            if "technology_skills" in job:
                for tech_skill in job["technology_skills"]:
                    if "skill_title" in tech_skill:
                        onet_skill_titles.add(tech_skill["skill_title"])
                    if "technologies" in tech_skill:
                        for tech_item in tech_skill["technologies"]:
                            if "name" in tech_item:
                                onet_tech_names.add(tech_item["name"])
    # else:
    #     # Dictionary format
    #     for job_title, job_data in onet_data.items():
    #         for field in ["technologies", "skills", "abilities"]:
    #             if field in job_data:
    #                 for item in job_data[field]:
    #                     if isinstance(item, dict) and "name" in item:
    #                         if field == "technologies":
    #                             onet_tech_names.add(item["name"])
    #                         else:
    #                             onet_skill_titles.add(item["name"])

    # Combine into a list of dictionaries
    onet_concepts = (
        [{"name": title, "type": "skill_title"} for title in onet_skill_titles] +
        [{"name": tech, "type": "technology_name"} for tech in onet_tech_names]
    )

    # Process each concept to separate the main text and the abbreviation
    processed_concepts = []
    for concept in onet_concepts:
        full_text = concept["name"]
        # Get the main part (everything before the first parenthesis)
        main_text = re.sub(r'\s*\(.*', '', full_text).strip()
        # Extract abbreviation if available
        abbr_match = re.search(r'\((.*?)\)', full_text)
        abbr_text = abbr_match.group(1).strip() if abbr_match else ""
        processed_concepts.append({
            "name": full_text,
            "type": concept["type"],
            "main": main_text,
            "abbr": abbr_text
        })

    # Generate embeddings if SentenceTransformer is available
    main_embeddings = None
    abbr_embeddings = None

    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Generating embeddings with SentenceTransformer...")
        # Initialize the model
        model_name = "sentence-transformers/msmarco-distilbert-base-v4"
        model = SentenceTransformer(model_name)

        # Create lists for the main texts and abbreviation texts
        main_texts = [item["main"] for item in processed_concepts]
        abbr_texts = [item["abbr"]
                      for item in processed_concepts if item["abbr"]]

        # Generate embeddings for main texts
        main_embeddings = model.encode(main_texts, convert_to_numpy=True)

        # Generate embeddings for abbreviations (if any)
        if abbr_texts:
            abbr_embeddings = model.encode(abbr_texts, convert_to_numpy=True)

        # Create a directory for saving embeddings if it doesn't exist
        os.makedirs("models", exist_ok=True)

        # Save embeddings and processed concepts for later use
        np.savez(f"models/onet_concept_embeddings_{model_name.replace('/', '_')}.npz",
                 main=main_embeddings, abbr=abbr_embeddings if abbr_embeddings is not None else np.array([]))
        with open(f"models/processed_onet_concepts_{model_name.replace('/', '_')}.json", "w", encoding="utf-8") as f:
            json.dump(processed_concepts, f, indent=4)

        print(f"Generated embeddings for {len(main_texts)} concepts")
        if abbr_embeddings is not None:
            print(f"Generated embeddings for {len(abbr_texts)} abbreviations")

    return processed_concepts, main_embeddings, abbr_embeddings

# Enhanced data preparation function


def prepare_data_for_gnn(data_path):
    """
    Prepare data for GNN model by constructing a graph from job and skill data.
    Modified to only include 'technology_name' and 'skill_title' from 'technology_skills'.

    Args:
        data_path (str): Path to the JSON data file

    Returns:
        tuple: Graph, node embeddings, adjacency matrix, nodes list, and node indices
    """
    import networkx as nx
    import json

    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Data type: {type(data)}")

    # Create graph
    G = nx.Graph()

    # Extract nodes and edges
    nodes = set()
    node_embeddings = {}

    # Default embedding dimension when generating random embeddings
    DEFAULT_EMBEDDING_DIM = 768  # Common embedding dimension

    # Track statistics
    skill_count = 0
    tech_count = 0
    job_count = 0

    # Enhanced extraction for 'technology_skills' only
    if isinstance(data, list):
        print(f"Processing list data with {len(data)} items")

        # First pass: extract all job titles
        for job_entry in data:
            if 'title' in job_entry:
                job_title = job_entry.get('title', '')
                if job_title:
                    G.add_node(job_title, node_type='job')
                    nodes.add(job_title)
                    job_count += 1

        # Second pass: extract 'technology_skills' and connect to jobs
        for job_entry in data:
            job_title = job_entry.get('title', '')
            if not job_title:
                continue

            # Extract from 'technology_skills'
            if "technology_skills" in job_entry:
                for tech_skill in job_entry["technology_skills"]:
                    # Add the skill title
                    if "skill_title" in tech_skill:
                        skill_name = tech_skill["skill_title"]
                        G.add_node(skill_name, node_type='skill_title')
                        nodes.add(skill_name)
                        G.add_edge(job_title, skill_name, weight=1.0)
                        skill_count += 1

                    # Add the technologies
                    if "technologies" in tech_skill:
                        for tech_item in tech_skill["technologies"]:
                            if "name" in tech_item:
                                tech_name = tech_item["name"]
                                G.add_node(
                                    tech_name, node_type='technology_name')
                                nodes.add(tech_name)
                                G.add_edge(job_title, tech_name,
                                           weight=tech_item.get("importance", 1.0))
                                tech_count += 1

    elif isinstance(data, dict):
        print(f"Processing dictionary data with {len(data.keys())} items")
        # Handle dictionary format
        for job_title, job_data in data.items():
            G.add_node(job_title, node_type='job')
            nodes.add(job_title)
            job_count += 1

            # Extract 'technology_skills'
            if "technology_skills" in job_data:
                for tech_skill in job_data["technology_skills"]:
                    # Add the skill title
                    if "skill_title" in tech_skill:
                        skill_name = tech_skill["skill_title"]
                        G.add_node(skill_name, node_type='skill_title')
                        nodes.add(skill_name)
                        G.add_edge(job_title, skill_name, weight=1.0)
                        skill_count += 1

                    # Add the technologies
                    if "technologies" in tech_skill:
                        for tech_item in tech_skill["technologies"]:
                            if "name" in tech_item:
                                tech_name = tech_item["name"]
                                G.add_node(
                                    tech_name, node_type='technology_name')
                                nodes.add(tech_name)
                                G.add_edge(job_title, tech_name,
                                           weight=tech_item.get("importance", 1.0))
                                tech_count += 1
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")

    print(
        f"Created graph with {len(nodes)} nodes and {G.number_of_edges()} edges")
    print(f"Job nodes: {job_count}")
    print(f"Skill title nodes: {skill_count}")
    print(f"Technology name nodes: {tech_count}")

    # Check if we have any nodes
    if len(nodes) == 0:
        raise ValueError("No nodes were created from the data")

    # If no embeddings found, generate random embeddings for all nodes
    if len(node_embeddings) == 0:
        print("No embeddings found in data. Generating random embeddings...")
        # Use a fixed seed for reproducibility
        np.random.seed(42)
        for node in nodes:
            node_embeddings[node] = np.random.randn(DEFAULT_EMBEDDING_DIM)
        print(
            f"Generated random embeddings with dimension {DEFAULT_EMBEDDING_DIM}")
    else:
        # Get embedding dimension from first embedding
        first_embedding = next(iter(node_embeddings.values()))
        embedding_dim = first_embedding.shape[0]
        print(f"Found embeddings with dimension {embedding_dim}")

        # Generate embeddings for nodes without embeddings
        missing_embeddings = nodes - set(node_embeddings.keys())
        if missing_embeddings:
            print(
                f"Generating embeddings for {len(missing_embeddings)} nodes without embeddings...")
            for node in missing_embeddings:
                node_embeddings[node] = np.random.randn(embedding_dim)

    # Create node index mapping
    nodes = list(nodes)
    node_indices = {node: i for i, node in enumerate(nodes)}

    # Create adjacency matrix
    num_nodes = len(nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for u, v, data in G.edges(data=True):
        i, j = node_indices[u], node_indices[v]
        weight = data.get('weight', 1.0)
        adj_matrix[i, j] = weight
        adj_matrix[j, i] = weight  # Undirected graph

    return G, node_embeddings, adj_matrix, nodes, node_indices

# Improved GNN Model with multi-head attention and skip connections


class ImprovedJobGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dims, num_classes, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate

        # GCN layer for initial feature processing
        self.conv1 = GCNConv(num_features, hidden_dims[0])

        # GAT layer with multi-head attention
        self.conv2 = GATConv(hidden_dims[0], hidden_dims[1] // 4, heads=4)

        # SAGE layer for neighborhood aggregation
        self.conv3 = SAGEConv(hidden_dims[1], hidden_dims[1])

        # Output layers with skip connection
        self.lin1 = torch.nn.Linear(
            hidden_dims[0] + hidden_dims[1], hidden_dims[1])
        self.lin2 = torch.nn.Linear(hidden_dims[1], num_classes)

        # Normalization and batch norm
        self.norm1 = torch.nn.LayerNorm(hidden_dims[0])
        self.norm2 = torch.nn.LayerNorm(hidden_dims[1])
        self.batch_norm = torch.nn.BatchNorm1d(hidden_dims[1])

        # Dropout for regularization
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        # First layer: Graph Convolutional Network
        x1 = self.conv1(x, edge_index)
        x1 = F.leaky_relu(x1)
        x1 = self.norm1(x1)
        x1 = self.dropout(x1)

        # Second layer: Graph Attention Network
        x2 = self.conv2(x1, edge_index)
        x2 = F.leaky_relu(x2)
        x2 = self.norm2(x2)
        x2 = self.dropout(x2)

        # Third layer: GraphSAGE with residual connection
        x3 = self.conv3(x2, edge_index)
        x3 = x3 + x2  # Residual connection
        x3 = self.batch_norm(x3)
        x3 = F.leaky_relu(x3)

        # Skip connection from first layer
        x_combined = torch.cat([x1, x3], dim=1)

        # Final prediction layers
        x_out = self.lin1(x_combined)
        x_out = F.leaky_relu(x_out)
        x_out = self.dropout(x_out)
        x_out = self.lin2(x_out)

        return x_out

# Advanced training with learning rate scheduling and early stopping


def train_improved_gnn(model, features, edge_index, mask, labels, device,
                       epochs=500, patience=50, lr=0.001, weight_decay=1e-4):
    """
    Train the GNN model with advanced techniques like early stopping and LR scheduling

    Args:
        model: The GNN model
        features: Node features tensor
        edge_index: Edge index tensor
        mask: Training mask tensor
        labels: Target labels tensor
        device: Computation device
        epochs: Maximum number of training epochs
        patience: Early stopping patience
        lr: Initial learning rate
        weight_decay: Weight decay for regularization

    Returns:
        Trained model
    """
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    # Class weighting for imbalanced data
    pos_count = labels.sum().item()
    neg_count = labels.shape[0]*labels.shape[1] - pos_count
    pos_weight = torch.tensor([neg_count / (pos_count + 1e-6)]).to(device)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training tracking variables
    best_loss = float('inf')  # Changed from best_score = 0
    best_epoch = 0
    no_improve = 0

    print("\nStarting training with early stopping based on loss:")
    for epoch in range(epochs):
        start_time = time.time()

        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(features, edge_index)
        loss = criterion(output[mask], labels.float())

        # Focused L2 regularization on final layer only
        l2_reg = 0.001 * model.lin2.weight.norm(2)
        loss += l2_reg

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Evaluation step
        model.eval()
        with torch.no_grad():
            val_output = model(features, edge_index)
            val_loss = criterion(val_output[mask], labels.float()).item()
            probs = torch.sigmoid(val_output[mask])

            # Calculate metrics for monitoring (not used for early stopping)
            top3_indices = torch.topk(probs, 3, dim=1).indices
            recall_at_3 = (torch.gather(labels, 1, top3_indices).sum(
                dim=1) > 0).float().mean()

            top5_indices = torch.topk(probs, 5, dim=1).indices
            recall_at_5 = (torch.gather(labels, 1, top5_indices).sum(
                dim=1) > 0).float().mean()

            # Learning rate scheduling based on validation loss
            scheduler.step(val_loss)  # Changed from current_score to val_loss

            # Early stopping check based on validation loss
            if val_loss < best_loss:  # Changed from current_score > best_score
                best_loss = val_loss  # Changed from best_score = current_score
                best_epoch = epoch
                torch.save(model.state_dict(), 'best_job_gnn_model.pt')
                no_improve = 0
            else:
                no_improve += 1

            # Print progress
            if (epoch+1) % 10 or epoch < 10:
                elapsed = time.time() - start_time
                print(f'Epoch {epoch+1}/{epochs} ({elapsed:.2f}s) | Loss: {val_loss:.4f} | '
                      f'Recall@3: {recall_at_3:.4f} | Recall@5: {recall_at_5:.4f} | '
                      f'Best Loss: {best_loss:.4f} @ {best_epoch}')

            # Early stopping
            if no_improve >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
                print(
                    f"Loading best model from epoch {best_epoch} with loss {best_loss:.4f}")
                model.load_state_dict(torch.load('best_job_gnn_model.pt',
                                                 map_location=device,
                                                 weights_only=True))
                break

    return model

# Enhanced prediction with domain-specific knowledge and better job title matching


def predict_jobs_improved(model, features, edge_index, skill_names, nodes, node_indices,
                          job_indices, top_k=10, skill_weights=None):
    """
    Make job predictions based on user skills with weighted similarity, domain knowledge,
    and enhanced job title matching

    Args:
        model: Trained GNN model
        features: Node features tensor
        edge_index: Edge index tensor
        skill_names: List of skill names
        nodes: List of all node names
        node_indices: Dictionary mapping node names to indices
        job_indices: List of job node indices
        top_k: Number of top jobs to return
        skill_weights: Optional weights for skills (based on user proficiency)

    Returns:
        List of (job name, score) tuples
    """
    model.eval()
    with torch.no_grad():
        # Get skill indices, handling missing skills gracefully
        skill_indices = []
        weights = []

        # Track skills for analyzing patterns
        found_skills = []
        skill_name_map = {}  # For domain identification

        for i, skill in enumerate(skill_names):
            if skill in node_indices:
                skill_indices.append(node_indices[skill])
                # Use provided weight or default to 1.0
                weight = 1.0
                if skill_weights and i < len(skill_weights):
                    weight = skill_weights[i]
                weights.append(weight)
                found_skills.append(skill)
                skill_name_map[skill.lower()] = weight

        if not skill_indices:
            return []

        # Convert to tensor - explicitly specify dtype=torch.float32
        weights = torch.tensor(weights, device=device, dtype=torch.float32)
        weights = weights / weights.sum()  # Normalize

        # Get node embeddings from the model's forward pass
        all_embeddings = model(features, edge_index)

        # Extract job and skill embeddings
        job_embeddings = all_embeddings[job_indices]
        skill_embeddings = all_embeddings[skill_indices]

        # Use weighted average based on skill importance
        weighted_skill_emb = torch.matmul(weights, skill_embeddings)

        # Calculate similarity scores with normalization
        job_emb_norm = F.normalize(job_embeddings, p=2, dim=1)
        skill_emb_norm = F.normalize(
            weighted_skill_emb.unsqueeze(0), p=2, dim=1)

        # Get cosine similarity scores
        scores = torch.matmul(job_emb_norm, skill_emb_norm.t()).squeeze()

        # Domain-specific knowledge: identify domains from skills
        # This helps boost relevant job titles
        domains = {
            "data_science": 0,
            "web_development": 0,
            "cloud_computing": 0,
            "cybersecurity": 0,
            "network_admin": 0,
            "database": 0,
            "machine_learning": 0,
            "mobile_dev": 0,
            "game_dev": 0,
            "devops": 0
        }

        # Map skills to domains
        data_science_skills = ["python", "r", "statistics", "statistical", "data analysis",
                               "machine learning", "deep learning", "pandas", "numpy",
                               "data visualization", "sql", "tableau", "power bi", "big data",
                               "data mining", "data modeling", "prediction", "forecasting",
                               "analytics", "business intelligence"]

        web_dev_skills = ["html", "css", "javascript", "react", "angular", "vue", "node.js",
                          "php", "ruby on rails", "django", "flask", "jquery", "bootstrap",
                          "web development", "front-end", "back-end", "full-stack"]

        cloud_skills = ["cloud", "aws", "azure", "google cloud", "kubernetes", "docker",
                        "containerization", "serverless", "microservices", "iaas", "paas",
                        "saas"]

        database_skills = ["sql", "mysql", "postgresql", "mongodb", "nosql", "oracle",
                           "database", "dbms", "data modeling", "etl"]

        ml_skills = ["machine learning", "deep learning", "neural networks", "tensorflow",
                     "pytorch", "keras", "nlp", "computer vision", "reinforcement learning"]

        # Calculate domain scores based on matched skills
        for skill, weight in skill_name_map.items():
            skill_lower = skill.lower()

            # Check for data science domain
            if any(ds_skill in skill_lower for ds_skill in data_science_skills):
                domains["data_science"] += weight

            # Check for web development domain
            if any(web_skill in skill_lower for web_skill in web_dev_skills):
                domains["web_development"] += weight

            # Check for cloud computing domain
            if any(cloud_skill in skill_lower for cloud_skill in cloud_skills):
                domains["cloud_computing"] += weight

            # Check for database domain
            if any(db_skill in skill_lower for db_skill in database_skills):
                domains["database"] += weight

            # Check for machine learning domain
            if any(ml_skill in skill_lower for ml_skill in ml_skills):
                domains["machine_learning"] += weight

        # Normalize domain scores
        total_domain_score = sum(domains.values()) + \
            1e-10  # Avoid division by zero
        for domain in domains:
            domains[domain] /= total_domain_score

        # Create domain boost tensor for job scores
        domain_boost = torch.ones_like(scores)

        # Dictionary mapping job keywords to domains for boosting
        job_keyword_to_domain = {
            # Data Science jobs
            "data scientist": "data_science",
            "data science": "data_science",
            "machine learning": "data_science",
            "data analyst": "data_science",
            "data engineer": "data_science",
            "business intelligence": "data_science",
            "statistician": "data_science",
            "analytics": "data_science",

            # Web Development jobs
            "web developer": "web_development",
            "frontend": "web_development",
            "backend": "web_development",
            "full stack": "web_development",
            "javascript": "web_development",
            "web designer": "web_development",

            # Cloud Computing jobs
            "cloud": "cloud_computing",
            "devops": "cloud_computing",
            "site reliability": "cloud_computing",
            "platform engineer": "cloud_computing",

            # Database jobs
            "database": "database",
            "dba": "database",
            "data architect": "database",

            # ML jobs
            "machine learning": "machine_learning",
            "ai engineer": "machine_learning",
            "ml engineer": "machine_learning",
            "nlp": "machine_learning",
            "computer vision": "machine_learning"
        }

        # Apply domain-specific boosts to job titles
        for i, job_idx in enumerate(job_indices):
            job_title = nodes[job_idx].lower()

            for keyword, domain in job_keyword_to_domain.items():
                if keyword in job_title:
                    # Boost jobs that match high-scoring domains
                    domain_boost[i] += domains[domain] * 0.5  # Up to 50% boost

        # Apply domain boost to the scores
        scores = scores * domain_boost

        # Handle specifically "Data Scientist" role with a special boost
        # if data science is the dominant domain
        if domains["data_science"] > 0.4:  # If 40% or more of the skills are data science related
            for i, job_idx in enumerate(job_indices):
                job_title = nodes[job_idx].lower()
                if "data scientist" in job_title or (
                        "data" in job_title and ("scientist" in job_title or "science" in job_title)):
                    # 30% boost specifically for Data Scientist
                    scores[i] *= 1.3

        # Filter out generic job titles with improved penalty
        generic_keywords = ['all other', 'miscellaneous', 'general',
                            'not elsewhere classified', 'occupations', 'various']

        # Create a filtration mask - lower score for generic job titles
        generality_penalty = torch.ones_like(scores)

        for i, job_idx in enumerate(job_indices):
            job_title = nodes[job_idx].lower()
            penalty_factor = 0.0

            for keyword in generic_keywords:
                if keyword in job_title:
                    penalty_factor += 0.2  # 20% penalty per generic keyword found

            if penalty_factor > 0:
                generality_penalty[i] = max(
                    0.4, 1.0 - penalty_factor)  # Cap penalty at 60%

        # Apply the penalty to scores
        scores = scores * generality_penalty

        # Give a boost to job titles that are more specific (contain multiple terms)
        specificity_boost = torch.ones_like(scores)
        for i, job_idx in enumerate(job_indices):
            job_title = nodes[job_idx]
            # Count terms (roughly by splitting on spaces, ignoring common words)
            terms = [t for t in job_title.lower().split() if len(
                t) > 3 and t not in ['and', 'the', 'with', 'for']]
            if len(terms) >= 3:  # More specific job titles tend to have more terms
                specificity_boost[i] = min(
                    1.5, 1.0 + (len(terms) - 2) * 0.1)  # Up to 50% boost

        # Apply the specificity boost
        scores = scores * specificity_boost

        # Analyze skill-job connections in the graph to boost highly relevant jobs
        skill_relevance_boost = torch.ones_like(scores)

        # Import networkx for graph analysis if available
        try:
            import networkx as nx
            has_networkx = True
        except ImportError:
            has_networkx = False

        if has_networkx and 'graph' in globals():
            G = globals()['graph']
            for i, job_idx in enumerate(job_indices):
                job_name = nodes[job_idx]

                # Calculate skill coverage - what percentage of the user's skills are connected to this job
                connected_skills = 0
                for skill in found_skills:
                    # Check if there's a path of length 1 between job and skill
                    if G.has_node(job_name) and G.has_node(skill) and skill in G.neighbors(job_name):
                        connected_skills += 1

                if found_skills:  # Avoid division by zero
                    coverage_ratio = connected_skills / len(found_skills)
                    # Boost jobs with higher skill coverage (up to 40% boost)
                    skill_relevance_boost[i] = 1.0 + (coverage_ratio * 0.4)

        # Apply the skill relevance boost
        scores = scores * skill_relevance_boost

        # Fix for negative scores: Convert cosine similarity range from [-1,1] to [0,1]
        scores = (scores + 1) / 2

        # Print score range for debugging
        print(
            f"Score range: min={scores.min().item():.4f}, max={scores.max().item():.4f}")

        # Get top predictions
        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))

    # Return job names and scores
    return [(nodes[job_indices[idx]], score.item()) for idx, score in zip(top_indices, top_scores)]

# Improved user input processing with better fuzzy matching


def process_user_input_improved(user_skills, nodes, graph):
    """
    Match user skills to existing graph nodes with advanced fuzzy matching

    Args:
        user_skills: List of (skill, type, proficiency) tuples from user input
        nodes: List of all node names in the graph
        graph: NetworkX graph with node_type attributes

    Returns:
        Tuple of (matched nodes list, weights list, rejected_matches dict)
    """
    matched_nodes = []
    weights = []
    skill_node_types = ['technology_name', 'skill_title', 'technology', 'skill',
                        'ability']  # Updated to include all possible skill node types

    # Track rejected matches for debugging
    rejected_matches = {
        "wrong_node_type": [],
        "below_threshold": [],
        "duplicate": []
    }

    # Track processed skill names to avoid duplicates
    processed_skills = set()

    # Create a mapping of lowercase node names to original names
    lower_nodes = [n.lower() for n in nodes]
    node_map = {n.lower(): n for n in nodes}

    # Also prepare a node_type map for easy lookup
    node_type_map = {}
    for node in nodes:
        try:
            node_type = graph.nodes[node].get('node_type', '')
            node_type_map[node.lower()] = node_type
        except Exception:
            continue

    print("\nAttempting to match user skills to knowledge graph nodes...")

    # First pass: try exact matches
    for skill_info in user_skills:
        skill, skill_type, proficiency = skill_info

        # Clean the skill name
        clean_skill = re.sub(r'[^a-zA-Z0-9 ]', '', skill).lower().strip()

        # Skip if we've already matched this skill (prevent duplicates)
        if clean_skill in processed_skills:
            continue

        print(f"Processing skill: '{skill}'")

        # Try exact match
        if clean_skill in node_map:
            original_node = node_map[clean_skill]
            node_type = node_type_map.get(clean_skill, '')

            # Check if this is a skill node, not a job node
            if node_type and node_type not in skill_node_types:
                print(
                    f"  ⨯ Found exact match '{original_node}' but it's a {node_type} node, not a skill node")
                rejected_matches["wrong_node_type"].append(
                    (skill, original_node, node_type))
                continue

            print(f"  ✓ Exact match: '{original_node}' ({node_type})")
            matched_nodes.append(original_node)
            weights.append(float(proficiency))
            processed_skills.add(clean_skill)
            continue

        # Extract keywords by removing parentheses and their content
        base_skill = re.sub(r'\([^)]*\)', '', clean_skill).strip()
        keywords = [k.strip() for k in base_skill.split() if len(k) > 2]

        # Try exact match with each keyword
        found = False
        for keyword in keywords:
            if keyword in node_map:
                original_node = node_map[keyword]
                node_type = node_type_map.get(keyword, '')

                # Check if this is a skill node, not a job node
                if node_type and node_type not in skill_node_types:
                    continue

                print(
                    f"  ✓ Keyword match: '{original_node}' via keyword '{keyword}'")
                matched_nodes.append(original_node)
                weights.append(float(proficiency))
                processed_skills.add(keyword)
                found = True
                break

        if found:
            continue

        # Fuzzy matching with higher threshold for better precision
        # Only match with skill nodes
        skill_nodes = [n.lower() for n in nodes if node_type_map.get(
            n.lower(), '') in skill_node_types]

        if not skill_nodes:
            print("  ⨯ No skill nodes found in the graph for fuzzy matching")
            continue

        matches = process.extract(clean_skill, skill_nodes, limit=3)
        matched = False
        for match_name, score in matches:
            # Increase threshold to 90 for more precise matching
            if score > 90:
                original_node = node_map[match_name]
                print(f"  ✓ Fuzzy match: '{original_node}' with score {score}")
                matched_nodes.append(original_node)
                # Weight by match quality and proficiency
                weights.append(float(proficiency) * (score / 100))
                processed_skills.add(match_name)
                matched = True
                break
            else:
                original_node = node_map[match_name]
                rejected_matches["below_threshold"].append(
                    (skill, original_node, score))

        if not matched:
            print(f"  ⨯ No good matches found for '{skill}'")

    # Print summary
    if matched_nodes:
        print(
            f"\nSuccessfully matched {len(matched_nodes)} out of {len(user_skills)} skills")
    else:
        print("\nWarning: Could not match any skills to the knowledge graph!")

    # Print rejected matches summary
    if any(rejected_matches.values()):
        print("\nRejected matches:")
        if rejected_matches["wrong_node_type"]:
            print(
                f"  - {len(rejected_matches['wrong_node_type'])} matches rejected due to wrong node type")
        if rejected_matches["below_threshold"]:
            print(
                f"  - {len(rejected_matches['below_threshold'])} matches rejected due to low similarity score")
        if rejected_matches["duplicate"]:
            print(
                f"  - {len(rejected_matches['duplicate'])} duplicates rejected")

    return matched_nodes, weights

# Parse user input with better error handling


def parse_user_input_improved(input_str):
    """Parse the user's skill input string into a list of tuples with better error handling"""
    import ast
    try:
        # Clean up the input string
        cleaned = re.sub(r'[\n\t]', '', input_str.strip())

        # Handle potential formatting issues
        if not cleaned.startswith('[') or not cleaned.endswith(']'):
            cleaned = f'[{cleaned}]'

        # Parse the string into a Python object
        parsed = ast.literal_eval(cleaned)

        # Convert to standardized format
        result = []
        for item in parsed:
            if len(item) >= 3:
                skill_name = item[0].strip()
                skill_type = item[1].strip()

                # Handle proficiency values
                try:
                    proficiency = float(item[2])
                    # Clamp to [0,1]
                    proficiency = max(0.0, min(1.0, proficiency))
                except (ValueError, TypeError):
                    proficiency = 1.0

                result.append((skill_name, skill_type, proficiency))

        return result

    except Exception as e:
        print(f"\nInput parsing error: {str(e)}")
        print("Please check your input format and try again.")
        return []

# Better results formatting with semantic grouping


def format_predictions_improved(predictions):
    """Format predictions for display with better layout and information"""
    if not predictions:
        return "No predictions available"

    # Sort by score descending
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    header = f"\n{'Job Title':<50} | {'Matching Score':<15}"
    divider = "-" * 70
    lines = [header, divider]

    for job, score in sorted_predictions:
        # Format score as percentage
        score_formatted = f"{score:.4f} ({score*100:.1f}%)"
        lines.append(f"{job:<50} | {score_formatted:<15}")

    return "\n".join(lines)

# Main execution function with improved embedding support


def run_job_prediction_pipeline(custom_input=None, use_sentence_transformer=True):
    """
    Run the complete job prediction pipeline with enhanced embedding support

    Args:
        custom_input (str, optional): Custom user input string
        use_sentence_transformer (bool): Whether to use sentence transformer embeddings

    Returns:
        tuple: Matched skills and predictions
    """
    # Initialize the predictions variable to avoid UnboundLocalError
    predictions = []

    # Use new paths for data and models
    data_path = os.path.join(DATA_PROCESSED_DIR, 'abbr_cleaned_IT_data_from_onet.json')
    model_save_path = os.path.join(MODEL_DIR, 'best_job_gnn_model.pt')
    embeddings_path = os.path.join(MODEL_DIR, 'onet_concept_embeddings_sentence-transformers_msmarco-distilbert-base-v4.npz')
    concepts_path = os.path.join(MODEL_DIR, 'processed_onet_concepts_sentence-transformers_msmarco-distilbert-base-v4.json')

    print(f"Running in Kaggle: {in_kaggle}")
    print(f"Using data path: {data_path}")

    # Process O*NET concepts with sentence transformers if enabled
    if use_sentence_transformer and SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n1. Generating embeddings using sentence transformers...")

        # Check if embeddings already exist
        if os.path.exists(embeddings_path) and os.path.exists(concepts_path):
            print(f"Loading existing embeddings from {embeddings_path}")
            # Load existing embeddings and concepts
            embeddings_data = np.load(embeddings_path)
            main_embeddings = embeddings_data['main']

            with open(concepts_path, 'r', encoding='utf-8') as f:
                processed_concepts = json.load(f)

            print(f"Loaded {len(processed_concepts)} concepts with embeddings")
        else:
            # Generate new embeddings
            print("Generating new embeddings for O*NET concepts...")
            processed_concepts, main_embeddings, _ = process_onet_concepts(
                data_path)
            print(
                f"Generated embeddings for {len(processed_concepts)} concepts")

        # Update node embeddings in the graph with these higher quality embeddings
        print("Applying embeddings to graph nodes...")

        # Build a lookup map from concept names to embeddings
        concept_to_embedding = {}
        for i, concept in enumerate(processed_concepts):
            concept_to_embedding[concept["name"]] = main_embeddings[i]
            # Also add the main part without abbreviation as a key
            concept_to_embedding[concept["main"]] = main_embeddings[i]
            # Add abbreviation if it exists
            if concept["abbr"]:
                concept_to_embedding[concept["abbr"]] = main_embeddings[i]

    # Load and prepare data
    print("\n2. Preparing data for GNN...")
    graph, node_embeddings, adj_matrix, nodes, node_indices = prepare_data_for_gnn(
        data_path)

    # Apply sentence transformer embeddings to node embeddings if available
    if use_sentence_transformer and SENTENCE_TRANSFORMERS_AVAILABLE and 'concept_to_embedding' in locals():
        print("Applying sentence transformer embeddings to node features...")
        # Get dimension from sentence transformer embeddings
        embedding_dim = main_embeddings.shape[1]
        updated_count = 0

        for node in nodes:
            if node in concept_to_embedding:
                node_embeddings[node] = concept_to_embedding[node]
                updated_count += 1

        print(
            f"Updated {updated_count} node embeddings with sentence transformer embeddings")

        # For nodes that didn't get a transformer embedding, generate a random one with the right dimension
        missing_count = 0
        for node in nodes:
            if node not in node_embeddings:
                node_embeddings[node] = np.random.randn(embedding_dim)
                missing_count += 1

        if missing_count > 0:
            print(f"Generated random embeddings for {missing_count} nodes")

    # Get embedding dimension and node counts
    embedding_dim = list(node_embeddings.values())[0].shape[0]
    job_indices = [idx for idx, node in enumerate(
        nodes) if graph.nodes[node].get('node_type') == 'job']
    skill_indices = [idx for idx, node in enumerate(
        nodes) if graph.nodes[node].get('node_type') in ['technology', 'skill', 'ability']]

    num_nodes = len(nodes)
    job_count = len(job_indices)
    skill_count = len(skill_indices)

    # Print node type breakdown for better understanding of the graph
    node_types = {}
    for node in nodes:
        node_type = graph.nodes[node].get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1

    print(f"\n3. Graph Statistics:")
    print(f"   - Total nodes: {num_nodes}")
    print(f"   - Job nodes: {job_count}")
    print(f"   - Skill nodes: {skill_count}")
    print(f"   - Node type breakdown:")
    for node_type, count in node_types.items():
        print(f"     * {node_type}: {count}")
    print(f"   - Total edges: {graph.number_of_edges()}")
    print(f"   - Embedding dimension: {embedding_dim}")

    # Prepare node features and adjacency data
    print("\n4. Preparing tensors...")
    features = np.zeros((num_nodes, embedding_dim))
    for node, idx in node_indices.items():
        if node in node_embeddings:
            features[idx] = node_embeddings[node]

    # Create multi-label targets for skill associations
    job_labels = np.zeros((job_count, num_nodes), dtype=np.float32)

    for job_idx, node_idx in enumerate(job_indices):
        neighbors = list(graph.neighbors(nodes[node_idx]))
        neighbor_indices = [node_indices[n]
                            for n in neighbors if n in node_indices]
        job_labels[job_idx, neighbor_indices] = 1.0

    # Convert to tensors
    features_tensor = torch.FloatTensor(features).to(device)
    adj_tensor = torch.FloatTensor(adj_matrix).to(device)
    edge_index, _ = add_self_loops(dense_to_sparse(adj_tensor)[0])
    job_mask_tensor = torch.BoolTensor(
        [i in job_indices for i in range(num_nodes)]).to(device)
    job_labels_tensor = torch.FloatTensor(job_labels).to(device)

    # Check if model already exists
    if in_kaggle:
        print("\n5. Initializing new model for Kaggle environment...")
        model = ImprovedJobGNN(
            num_features=embedding_dim,
            hidden_dims=[1024, 512],
            num_classes=num_nodes,
            dropout_rate=0.3
        ).to(device)

        # Train with fewer epochs in Kaggle to avoid timeout
        print("\n6. Training model in Kaggle (with reduced epochs)...")
        model = train_improved_gnn(
            model,
            features_tensor,
            edge_index,
            job_mask_tensor,
            job_labels_tensor,
            device,
            epochs=200,  # Reduced epochs for Kaggle
            patience=40,  # Reduced patience for Kaggle
            lr=0.001,
            weight_decay=1e-4
        )

        print(f"\n7. Saving trained model to {model_save_path}...")
        torch.save(model.state_dict(), model_save_path)

    else:
        # Regular local environment handling
        model_path = 'best_job_gnn_model.pt'
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            print(f"\n5. Loading pre-trained model from {model_path}...")
            model = ImprovedJobGNN(
                num_features=embedding_dim,
                hidden_dims=[1024, 512],
                num_classes=num_nodes,
                dropout_rate=0.3
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            # Initialize and train model
            print("\n5. Initializing model...")
            model = ImprovedJobGNN(
                num_features=embedding_dim,
                hidden_dims=[1024, 512],
                num_classes=num_nodes,
                dropout_rate=0.3
            ).to(device)

            print("\n6. Training model (this may take a while)...")
            model = train_improved_gnn(
                model,
                features_tensor,
                edge_index,
                job_mask_tensor,
                job_labels_tensor,
                device,
                epochs=2000,
                patience=100,
                lr=0.001,
                weight_decay=1e-4
            )

            print("\n7. Saving trained model...")
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), 'models/improved_job_predictor.pt')

    # Enhanced matching with sentence transformer if enabled
    if use_sentence_transformer and SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n8. Using sentence transformer for enhanced skill matching...")

        # Load the sentence transformer model for user input processing
        model_name = "sentence-transformers/msmarco-distilbert-base-v4"
        st_model = SentenceTransformer(model_name)

        # Function to match user skill to graph node with semantic similarity
        def match_skills_with_transformer(user_skills, nodes, graph, st_model, concept_to_embedding):
            matched_nodes = []
            weights = []
            # Update this to match the actual node types in your graph
            skill_node_types = ['technology_name', 'skill_title']

            print("\nMatching skills using semantic similarity...")

            # Create a mapping of nodes that are skills (not jobs)
            skill_nodes = [n for n in nodes if graph.nodes[n].get(
                'node_type', '') in skill_node_types]

            if not skill_nodes:
                print("No skill nodes found in graph")
                return [], []

            # Generate embeddings for all skill nodes (cache)
            skill_node_embeddings = {}
            for node in skill_nodes:
                if node in concept_to_embedding:
                    skill_node_embeddings[node] = concept_to_embedding[node]

            # Process each user skill
            for skill_info in user_skills:
                skill, skill_type, proficiency = skill_info
                print(f"Processing skill: '{skill}'")

                # Generate embedding for user skill
                user_skill_embedding = st_model.encode(
                    [skill], convert_to_numpy=True)[0]

                # Calculate similarity with all skill nodes
                best_score = -1
                best_node = None

                for node in skill_nodes:
                    if node in skill_node_embeddings:
                        node_emb = skill_node_embeddings[node]
                        # Calculate cosine similarity
                        similarity = cosine_similarity(
                            [user_skill_embedding], [node_emb])[0][0]

                        if similarity > best_score and similarity > 0.7:  # Threshold for good match
                            best_score = similarity
                            best_node = node

                # Add the best match if found
                if best_node:
                    print(
                        f"  ✓ Semantic match: '{best_node}' with score {best_score:.4f}")
                    matched_nodes.append(best_node)
                    # Weight by similarity score and proficiency
                    weights.append(float(proficiency) * best_score)
                else:
                    print(f"  ⨯ No good semantic matches found for '{skill}'")

            return matched_nodes, weights

        # Now for user input processing, we'll use both methods
        # Parse the user input
        user_skills = parse_user_input_improved(custom_input if custom_input is not None else """
        [
    ("Python", "technology_name", 1.00),
    ("Amazon Web Services software (AWS)", "technology_name", 1.00),
    ("Structured query language (SQL)", "technology_name", 1.00),
    ("Microsoft Azure software (Azure)", "technology_name", 1.00),
    ("Tableau", "technology_name", 1.00),
    ("Business intelligence software", "skill_title", 1.00),
    ("data analysis software", "skill_title", 1.00),
    ("Development environment software (Dev Env SW)", "skill_title", 1.00),
    ("Operating system software (OS SW)", "skill_title", 1.00),
    ("Other Technology Skills (Other Tech SW)", "skill_title", 1.00),
]
        """)

        print(f"\n9. Parsed {len(user_skills)} skills from user input")

        # Try semantic matching first
        matched_skills, skill_weights = match_skills_with_transformer(
            user_skills, nodes, graph, st_model, concept_to_embedding
        )

        # If semantic matching didn't find enough matches, fall back to fuzzy matching
        if len(matched_skills) < len(user_skills) * 0.5:
            print("\nFalling back to fuzzy matching for remaining skills...")
            fuzzy_matched, fuzzy_weights = process_user_input_improved(
                user_skills, nodes, graph
            )

            # Combine the results (avoiding duplicates)
            existing = set(matched_skills)
            for i, skill in enumerate(fuzzy_matched):
                if skill not in existing:
                    matched_skills.append(skill)
                    skill_weights.append(fuzzy_weights[i])
                    existing.add(skill)
    else:
        # Use the original fuzzy matching approach
        print("\n8. Running prediction...")

        # Use custom input if provided, otherwise use a default sample
        if custom_input is not None:
            sample_input = custom_input
        else:
            # Default sample input as fallback
            sample_input = """
            [
    ("Python", "technology_name", 1.00),
    ("Amazon Web Services software (AWS)", "technology_name", 1.00),
    ("Structured query language (SQL)", "technology_name", 1.00),
    ("Microsoft", "technology_name", 1.00),
    ("Web server software (Web Server)", "technology_name", 0.80),
    ("User interface design software (UI Design)", "technology_name", 0.78),
    ("Business intelligence software", "skill_title", 0.74),
    ("Graphical user interface GUI design software (GUI Design)", "technology_name", 0.72),
    ("data analysis software", "skill_title", 0.71),
    ("Graphical user interface development software (GUI Dev SW)", "skill_title", 0.69),
    ("analysis software", "skill_title", 0.67),
    ("Software development tools (Dev Tools)", "technology_name", 0.63),
    ("Amazon Web Services CloudFormation (CloudFormation)", "technology_name", 0.63),
    ("design software", "skill_title", 0.62),
    ("Platform as a service PaaS (PaaS)", "technology_name", 0.60),
    ("Development environment software (Dev Env SW)", "skill_title", 0.59),
    ("Citrix cloud computing software (Citrix Cloud)", "technology_name", 0.58),
    ("Other Technology Skills (Other Tech SW)", "skill_title", 0.55),
    ("IBM InfoSphere DataStage (DataStage)", "technology_name", 0.52),
    ("UserZoom", "technology_name", 0.51),
]

            """

        # Parse and process user input
        user_skills = parse_user_input_improved(sample_input)
        print(f"\n9. Parsed {len(user_skills)} skills from user input")

        # Pass the graph to the improved matching function
        matched_skills, skill_weights = process_user_input_improved(
            user_skills, nodes, graph)

    # Generate predictions with matched skills
    print(
        f"\n10. Matched {len(matched_skills)} skills to knowledge graph nodes")

    # Print the matched skills for transparency
    if matched_skills:
        print("\nMatched skills:")
        for i, skill in enumerate(matched_skills):
            print(f"  - {skill} (weight: {skill_weights[i]:.2f})")

        predictions = predict_jobs_improved(
            model=model,
            features=features_tensor,
            edge_index=edge_index,
            skill_names=matched_skills,
            nodes=nodes,
            node_indices=node_indices,
            job_indices=job_indices,
            top_k=10,
            skill_weights=skill_weights
        )

        print("\n11. Top Job Predictions:")
        print(format_predictions_improved(predictions))
    else:
        print("\nNo matching skills found in knowledge graph")

    print("\nJob prediction complete!")

    return matched_skills, predictions  # Return results for further analysis


# Execute the pipeline if this script is run directly
if __name__ == "__main__":
    # When running as a script, use the default sample
    run_job_prediction_pipeline()
