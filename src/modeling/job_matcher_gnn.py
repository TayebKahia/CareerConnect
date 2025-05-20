# This file should be moved to src/modeling/ as it is a modeling-related script.
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.helpers import debug_log

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'job_matcher_gnn.pt')

# Load the ONET data


def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Data Preprocessing Functions


def extract_job_skills_data(data):
    job_data = []

    for job in data:
        # Extract job code and title
        job_code = job.get("code", "")
        job_title = job.get("title", "Unknown Job Title")

        # Extract technology skills and technologies
        skills_data = []
        tech_skills = job.get("technology_skills", [])

        for skill_group in tech_skills:
            skill_title = skill_group.get("skill_title", "")
            technologies = skill_group.get("technologies", [])

            # Add skill title
            if skill_title:
                skills_data.append({
                    "name": skill_title,
                    "type": "skill_title",
                    "weight": 1.0
                })

            # Add technologies
            for tech in technologies:
                tech_name = tech.get("name", "")
                if tech_name:
                    skills_data.append({
                        "name": tech_name,
                        "type": "technology_name",
                        "weight": 1.0
                    })

        # Only include jobs with at least one skill or technology
        if skills_data:
            job_data.append({
                "code": job_code,
                "title": job_title,
                "skills": skills_data
            })

    return job_data

# Function to build graph data with job-specific identifiers


def build_graph_data(job_data, concept_embeddings, concept_to_idx, job_to_idx):
    graph_data_list = []
    job_to_skills_map = {}  # Map to track skills for each job

    print("Building graph data...")
    for job in tqdm(job_data):
        job_title = job["title"]
        job_idx = job_to_idx[job_title]
        job_skills = job["skills"]

        # Skip jobs with no skills
        if not job_skills:
            continue

        # Track skill names for this job
        skill_names = []

        # Create node features from skill embeddings
        skill_indices = []
        for skill in job_skills:
            skill_name = skill["name"]
            if skill_name in concept_to_idx:
                skill_indices.append(concept_to_idx[skill_name])
                skill_names.append(skill_name)

        # Store the skills for this job
        job_to_skills_map[job_title] = skill_names

        # Skip if no skills matched
        if not skill_indices:
            continue

        # Get node features
        node_features = concept_embeddings[skill_indices]

        # Create a complete graph (all skills are connected to each other)
        num_nodes = len(node_features)
        edge_indices = []

        # Connect skills to each other (complete graph)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_indices.append([i, j])

        if not edge_indices:
            # If no edges, create self-loops
            edge_indices = [[i, i] for i in range(num_nodes)]

        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(
            edge_indices, dtype=torch.long).t().contiguous()
        y = torch.tensor([job_idx], dtype=torch.long)

        # Create graph data object
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )

        graph_data_list.append(graph_data)

    return graph_data_list, job_to_skills_map

# Define a more powerful GNN model for overfitting


class OverfitJobGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(OverfitJobGNN, self).__init__()
        # Increase model capacity
        self.hidden_dim = hidden_dim

        # Multiple graph convolution layers for better feature extraction
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Attention layers for better node importance weighting
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)

        # Final classification layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        # If batch is None (single graph), create a batch tensor
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Deep feature extraction
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Apply attention to focus on important nodes
        x = self.gat1(x, edge_index)
        x = F.relu(x)

        # Global pooling - use both mean and add pooling for better graph representation
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x = x_mean + x_add

        # Final classification
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

# Training function with higher learning rate and no regularization to promote overfitting


def train_overfit_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)

# Evaluation function


def evaluate_model(model, loader, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())

    return correct / len(loader.dataset)

# Function to predict job title from user input with confidence score


def predict_job_title(user_input, model, embedding_model, concept_to_idx, idx_to_job, job_to_skills_map, device):
    model.eval()

    # Process user input and create graph
    node_names = []
    node_types = []
    node_weights = []

    for item in user_input:
        name, item_type, weight = item
        if name in concept_to_idx:
            node_names.append(name)
            node_types.append(item_type)
            node_weights.append(weight)
        else:
            print(f"Warning: '{name}' not found in training data vocabulary.")

    # Skill matching analysis
    job_match_scores = {}
    for job_title, job_skills in job_to_skills_map.items():
        # Calculate what percentage of this job's skills are in the user input
        matched_skills = set(node_names).intersection(set(job_skills))
        if len(job_skills) > 0:
            match_score = len(matched_skills) / len(job_skills)
            job_match_scores[job_title] = (
                match_score, len(matched_skills), len(job_skills))

    # Print skill matching statistics
    print("\nSkill Matching Analysis:")
    top_matches = sorted(job_match_scores.items(),
                         key=lambda x: x[1][0], reverse=True)[:3]
    for job, (score, matched, total) in top_matches:
        print(f"- {job}: {score*100:.1f}% match ({matched}/{total} skills)")

    # Skip if no valid skills/technologies
    if not node_names:
        return "No matching skills/technologies found in the model vocabulary. Please check your input.", None

    print(
        f"Found {len(node_names)} matching skills/technologies in the model vocabulary.")

    # Generate embeddings for the nodes
    embeddings = embedding_model.encode(node_names, convert_to_tensor=True)

    # Apply weights to embeddings (element-wise multiplication)
    weights_tensor = torch.tensor(
        node_weights, dtype=torch.float, device=embeddings.device).view(-1, 1)
    weighted_embeddings = embeddings * weights_tensor

    # Create a complete graph connecting all nodes
    num_nodes = len(weighted_embeddings)
    edge_indices = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_indices.append([i, j])

    # If no edges, create self-loops
    if not edge_indices:
        edge_indices = [[i, i] for i in range(num_nodes)]

    # Create tensors
    x = weighted_embeddings
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Move to device
    x = x.to(device)
    edge_index = edge_index.to(device)

    # Make prediction
    with torch.no_grad():
        # For single graph (no batch)
        batch = None
        out = model(x, edge_index, batch)
        probabilities = torch.exp(out)

        # Get all predictions with their probabilities
        probs, indices = torch.sort(probabilities[0], descending=True)

    # Return all predictions with meaningful confidence
    results = []
    for i in range(min(5, len(probs))):
        job_idx = indices[i].item()
        prob = probs[i].item()
        job_title = idx_to_job[job_idx]
        results.append((job_title, prob))

    return results, job_match_scores


def main():
    # 1. Load data
    print("Loading data...")
    file_path = os.path.join(DATA_PROCESSED_DIR, 'abbr_cleaned_IT_data_from_onet.json')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the ONET data file at {file_path}. Please check the file path.")
    data = load_json_data(file_path)

    # 2. Extract job skills data
    print("Extracting job skills data...")
    job_data = extract_job_skills_data(data)
    print(f"Found {len(job_data)} jobs with skills data")

    # 3. Prepare concepts (skills and technologies)
    all_concepts = set()
    job_titles = []

    for job in job_data:
        job_titles.append(job["title"])
        for skill in job["skills"]:
            all_concepts.add(skill["name"])

    concept_list = list(all_concepts)
    job_titles = list(set(job_titles))

    print(f"Found {len(concept_list)} unique skill/technology concepts")
    print(f"Found {len(job_titles)} unique job titles")

    # 4. Create mappings
    concept_to_idx = {concept: idx for idx, concept in enumerate(concept_list)}
    job_to_idx = {job: idx for idx, job in enumerate(job_titles)}
    idx_to_job = {idx: job for job, idx in job_to_idx.items()}

    # 5. Generate concept embeddings
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(
        'sentence-transformers/msmarco-distilbert-base-v4')
    embedding_model = embedding_model.to(device)

    print("Generating concept embeddings...")
    concept_embeddings = embedding_model.encode(
        concept_list, show_progress_bar=True, convert_to_tensor=True).cpu().numpy()

    # 6. Build graph data
    graph_data_list, job_to_skills_map = build_graph_data(
        job_data, concept_embeddings, concept_to_idx, job_to_idx)
    print(f"Built {len(graph_data_list)} graph data objects")

    # 7. Use all data for training (intentional overfitting)
    train_loader = DataLoader(graph_data_list, batch_size=16, shuffle=True)

    # 8. Initialize model with high capacity
    input_dim = concept_embeddings.shape[1]  # Embedding dimension
    hidden_dim = 256  # Larger hidden dimension
    output_dim = len(job_titles)

    model = OverfitJobGNN(input_dim, hidden_dim, output_dim).to(device)

    # Higher learning rate to promote faster convergence/overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 9. Train model
    print("\nTraining model (intentionally overfitting)...")
    num_epochs = 200  # Train for many epochs
    train_accs = []
    losses = []

    for epoch in range(1, num_epochs + 1):
        loss = train_overfit_model(model, train_loader, optimizer, device)
        train_acc = evaluate_model(model, train_loader, device)

        losses.append(loss)
        train_accs.append(train_acc)

        if epoch % 10 == 0:
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}')

        # Stop if perfect accuracy is achieved
        if train_acc > 0.99:
            print(
                f"Perfect training accuracy achieved at epoch {epoch}. Stopping training.")
            break

    # 10. Evaluate final model
    final_train_acc = evaluate_model(model, train_loader, device)
    print(f'\nFinal Train Accuracy: {final_train_acc:.4f}')

    # Plot training progress
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(
        MODEL_SAVE_PATH), 'training_progress.png')
    plt.savefig(plot_path)
    print(f"Training progress plot saved to {plot_path}")

    # 11. Save the model and data
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim
        }, MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

        model_data_path = os.path.join(os.path.dirname(
            MODEL_SAVE_PATH), 'job_matcher_data.json')

        # Save only essential data using standard Python types
        serializable_concept_to_idx = {
            k: int(v) for k, v in concept_to_idx.items()}
        serializable_job_to_idx = {k: int(v) for k, v in job_to_idx.items()}
        serializable_idx_to_job = {str(k): v for k, v in idx_to_job.items()}

        with open(model_data_path, 'w') as f:
            json.dump({
                "concept_to_idx": serializable_concept_to_idx,
                "job_to_idx": serializable_job_to_idx,
                "idx_to_job": serializable_idx_to_job,
                "final_train_accuracy": float(final_train_acc)
            }, f)
        print(f"Model data saved to {model_data_path}")

        # Save job to skills map
        skills_map_path = os.path.join(os.path.dirname(
            MODEL_SAVE_PATH), 'job_skills_map.json')
        with open(skills_map_path, 'w') as f:
            json.dump(job_to_skills_map, f)
        print(f"Job skills map saved to {skills_map_path}")

    except Exception as e:
        print(f"Error saving model: {e}")

    # 12. Test with a sample user input
    sample_user_input = [
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

    print("\nPredicting job title for sample input...")
    predictions, match_scores = predict_job_title(
        sample_user_input,
        model,
        embedding_model,
        concept_to_idx,
        idx_to_job,
        job_to_skills_map,
        device
    )

    print("\nTop job predictions:")
    for i, (job, prob) in enumerate(predictions):
        print(f"{i+1}. {job} (Confidence: {prob:.4f})")

# Create a simple interface for job title prediction


def predict_with_saved_model(user_input):
    """
    Uses a saved model to predict job titles based on skills/technologies

    Args:
        user_input: List of tuples (skill/tech name, type, weight)
                   Example: [("Python", "technology_name", 1.00), ...]

    Returns:
        List of top job predictions with confidence scores
    """
    # Load saved model and data
    model_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '..', '..', 'models')
    model_path = os.path.join(model_dir, 'job_matcher_gnn.pt')
    data_path = os.path.join(model_dir, 'job_matcher_data.json')
    skills_map_path = os.path.join(model_dir, 'job_skills_map.json')

    # Check if files exist
    if not (os.path.exists(model_path) and os.path.exists(data_path)):
        return "Model files not found. Please train the model first."

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model data
    with open(data_path, 'r') as f:
        model_data = json.load(f)

    concept_to_idx = model_data["concept_to_idx"]
    idx_to_job = {int(k): v for k, v in model_data["idx_to_job"].items()}

    # Load job skills map
    with open(skills_map_path, 'r') as f:
        job_to_skills_map = json.load(f)

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    input_dim = checkpoint['input_dim']
    hidden_dim = checkpoint['hidden_dim']
    output_dim = checkpoint['output_dim']

    model = OverfitJobGNN(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load embedding model
    embedding_model = SentenceTransformer(
        'sentence-transformers/msmarco-distilbert-base-v4')
    embedding_model = embedding_model.to(device)

    # Make prediction
    predictions, match_scores = predict_job_title(
        user_input,
        model,
        embedding_model,
        concept_to_idx,
        idx_to_job,
        job_to_skills_map,
        device
    )

    return predictions, match_scores


if __name__ == "__main__":
    main()
