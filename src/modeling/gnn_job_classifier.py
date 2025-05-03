# This file should be moved to src/modeling/ as it is a modeling-related script.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import json
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
# If you need helpers or debug_log, import from src.utils.helpers
# from src.utils.helpers import debug_log

# Define project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

class JobClassifierGNN(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=64):
        super(JobClassifierGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x, edge_index, batch):
        # First Graph Convolution Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second Graph Convolution Layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def prepare_graph_data(graph, skill_embeddings, job_labels):
    """
    Prepare graph data for PyTorch Geometric
    """
    # Convert NetworkX graph to PyTorch Geometric format
    edge_index = []
    for u, v in graph.edges():
        edge_index.append([u, v])
        edge_index.append([v, u])  # Add reverse edge for undirected graph
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Node features (using skill embeddings)
    x = torch.tensor(skill_embeddings, dtype=torch.float)
    
    # Node labels (job classifications)
    y = torch.tensor(job_labels, dtype=torch.long)
    
    return x, edge_index, y

def train_model(model, optimizer, x, edge_index, y, batch_size=32, epochs=100):
    """
    Train the GNN model
    """
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        out = model(x, edge_index, torch.zeros(x.size(0), dtype=torch.long))
        loss = F.nll_loss(out, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

def predict_job(model, skill_set, skill_embeddings, graph):
    """
    Predict job classification for a given set of skills
    """
    model.eval()
    with torch.no_grad():
        # Convert input skills to embeddings
        input_embeddings = torch.tensor(skill_embeddings, dtype=torch.float)
        
        # Get graph structure
        edge_index = []
        for u, v in graph.edges():
            edge_index.append([u, v])
            edge_index.append([v, u])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Make prediction
        out = model(input_embeddings, edge_index, torch.zeros(input_embeddings.size(0), dtype=torch.long))
        pred = out.argmax(dim=1)
        
        return pred

def main():
    # Load the graph and embeddings from the previous notebook
    with open(os.path.join(DATA_PROCESSED_DIR, "processed_onet_concepts_sentence-transformers_msmarco-distilbert-base-v4.json"), "r") as f:
        processed_concepts = json.load(f)
    
    # Load the graph
    G = nx.read_gpickle("onet_similarity_graph.gpickle")
    
    # Load embeddings
    embeddings_data = np.load("onet_concept_embeddings_sentence-transformers_msmarco-distilbert-base-v4.npz")
    main_embeddings = embeddings_data['main']
    
    # Prepare job labels (you'll need to modify this based on your actual job data)
    # This is a placeholder - you should replace this with actual job labels
    job_labels = np.zeros(len(G.nodes))  # Replace with actual job labels
    
    # Initialize the model
    num_node_features = main_embeddings.shape[1]
    num_classes = len(np.unique(job_labels))
    model = JobClassifierGNN(num_node_features, num_classes)
    
    # Prepare data
    x, edge_index, y = prepare_graph_data(G, main_embeddings, job_labels)
    
    # Split data into train and test sets
    train_idx, test_idx = train_test_split(range(len(y)), test_size=0.2, random_state=42)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train the model
    train_model(model, optimizer, x[train_idx], edge_index, y[train_idx])
    
    # Example prediction
    test_skills = ["Python", "Machine Learning", "Data Analysis"]
    prediction = predict_job(model, test_skills, main_embeddings, G)
    print(f"Predicted job classification: {prediction}")

if __name__ == "__main__":
    main() 