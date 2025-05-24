"""
Graph Neural Network wrapper for job role prediction.
This adapts the existing GNN implementation for use in the Hybrid Ensemble model.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import joblib

# Add parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the existing GNN model implementation
from modeling.gnn_job_classifier import JobClassifierGNN

# Define project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Technology columns in the dataset
TECH_COLUMNS = [
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "PlatformHaveWorkedWith",
    "WebframeHaveWorkedWith", 
    "MiscTechHaveWorkedWith",
    "ToolsTechHaveWorkedWith",
    "EmbeddedHaveWorkedWith"
]

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GNNJobClassifier:
    """
    Wrapper for the GNN-based job classifier.
    This adapts the existing GNN implementation for use in the ensemble.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the GNN classifier.
        
        Args:
            model_path: Path to load a pre-trained model from.
        """
        self.model = None
        self.graph = None
        self.data = None
        self.tech_to_node = {}
        self.node_to_tech = {}
        self.label_encoder = None
        self.node_embeddings = None
        
        if model_path:
            self.load_model(model_path)
    
    def build_graph(self, data):
        """
        Build a graph from the dataset for GNN training.
        
        Args:
            data: DataFrame containing DevType and technology columns
            
        Returns:
            graph: NetworkX graph
            tech_to_node: Mapping from technology name to node ID
            node_to_tech: Mapping from node ID to technology name
        """
        G = nx.Graph()
        
        # Process technologies from all columns
        all_techs = set()
        for col in TECH_COLUMNS:
            if col in data.columns:
                # Extract technologies from each row
                for techs in data[col].dropna():
                    if techs:
                        for tech in techs.split(';'):
                            tech = tech.strip()
                            if tech:
                                all_techs.add(tech)
        
        # Create nodes for technologies
        for i, tech in enumerate(sorted(all_techs)):
            G.add_node(i, type='tech', name=tech, embedding=None)
            self.tech_to_node[tech] = i
            self.node_to_tech[i] = tech
        
        # Encode job roles
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(data['DevType'])
        
        # Create job role nodes and connect technologies based on co-occurrence
        tech_end_idx = len(self.tech_to_node)
        job_start_idx = tech_end_idx
        
        for job_id, job_name in enumerate(self.label_encoder.classes_):
            node_id = job_start_idx + job_id
            G.add_node(node_id, type='job', name=job_name, job_id=job_id)
        
        # Add edges based on technology co-occurrence in job roles
        for _, row in data.iterrows():
            job_id = self.label_encoder.transform([row['DevType']])[0]
            job_node_id = job_start_idx + job_id
            
            # Connect technologies to job roles
            row_techs = []
            for col in TECH_COLUMNS:
                if col in data.columns and pd.notna(row[col]):
                    for tech in row[col].split(';'):
                        tech = tech.strip()
                        if tech:
                            row_techs.append(tech)
            
            # Add edges between the job and each technology
            for tech in row_techs:
                if tech in self.tech_to_node:
                    tech_node_id = self.tech_to_node[tech]
                    G.add_edge(tech_node_id, job_node_id, weight=1.0)
            
            # Add edges between co-occurring technologies
            for i, tech1 in enumerate(row_techs):
                if tech1 in self.tech_to_node:
                    tech1_node_id = self.tech_to_node[tech1]
                    for tech2 in row_techs[i+1:]:
                        if tech2 in self.tech_to_node:
                            tech2_node_id = self.tech_to_node[tech2]
                            if G.has_edge(tech1_node_id, tech2_node_id):
                                G[tech1_node_id][tech2_node_id]['weight'] += 1.0
                            else:
                                G.add_edge(tech1_node_id, tech2_node_id, weight=1.0)
        
        # Normalize edge weights
        max_weight = max([d['weight'] for _, _, d in G.edges(data=True)], default=1.0)
        for u, v, d in G.edges(data=True):
            d['weight'] = d['weight'] / max_weight
        
        return G
    
def prepare_pytorch_geometric_data(self, graph):
        """
        Convert NetworkX graph to PyTorch Geometric Data object.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            data: PyTorch Geometric Data object
        """
        # Extract node features as simple one-hot embeddings for tech vs job
        num_nodes = graph.number_of_nodes()
        node_features = np.zeros((num_nodes, 2), dtype=np.float32)
        node_labels = np.zeros(num_nodes, dtype=np.int64)
        
        for node_id, node_data in graph.nodes(data=True):
            if node_data['type'] == 'tech':
                node_features[node_id, 0] = 1.0
            else:  # job
                node_features[node_id, 1] = 1.0
                node_labels[node_id] = node_data.get('job_id', 0)
        
        # Extract edges and weights
        edge_index = []
        edge_attr = []
        
        for u, v, data in graph.edges(data=True):
            edge_index.append([u, v])
            edge_index.append([v, u])  # Add reverse edge for undirected graph
            weight = data.get('weight', 1.0)
            edge_attr.append([weight])
            edge_attr.append([weight])  # Same weight for reverse edge
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        node_features = torch.tensor(node_features, dtype=torch.float)
        node_labels = torch.tensor(node_labels, dtype=torch.long)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=node_labels
        )
        
        return data
    
def train(self, data_path, hidden_channels=64, epochs=100, lr=0.01):
        """
        Train the GNN classifier on the dataset.
        
        Args:
            data_path: Path to the CSV file containing the dataset
            hidden_channels: Number of hidden channels in GNN
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Load and preprocess data
        data = pd.read_csv(data_path)
        
        # Process technology columns
        for col in TECH_COLUMNS:
            if col in data.columns:
                data[col] = data[col].fillna('')
        
        # Build graph
        print("Building graph from dataset...")
        self.graph = self.build_graph(data)
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        # Convert to PyTorch Geometric Data
        self.data = self.prepare_pytorch_geometric_data(self.graph)
        print("Converted graph to PyTorch Geometric format")
        
        # Split data into train and test sets (by masking node labels)
        num_nodes = self.data.x.size(0)
        num_techs = len(self.tech_to_node)
        num_jobs = len(self.label_encoder.classes_)
        job_node_indices = torch.arange(num_techs, num_nodes)
        
        # Random permutation of job nodes
        perm = torch.randperm(len(job_node_indices))
        job_node_indices = job_node_indices[perm]
        
        # Split into train/val/test
        train_idx = job_node_indices[:int(0.6 * len(job_node_indices))]
        val_idx = job_node_indices[int(0.6 * len(job_node_indices)):int(0.8 * len(job_node_indices))]
        test_idx = job_node_indices[int(0.8 * len(job_node_indices)):]
        
        # Create train/val/test masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        self.data.train_mask = train_mask
        self.data.val_mask = val_mask
        self.data.test_mask = test_mask
        
        # Move data to device
        self.data = self.data.to(DEVICE)
        
        # Initialize model
        num_node_features = self.data.x.size(1)
        num_classes = num_jobs
        
        self.model = JobClassifierGNN(
            num_node_features=num_node_features,
            num_classes=num_classes,
            hidden_channels=hidden_channels
        ).to(DEVICE)
        
        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        best_val_acc = 0
        best_model_state = None
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index, None)
            
            # Reshape tensors to match dimensions
            # Get only the output for nodes that are in the training mask
            train_mask_indices = torch.where(self.data.train_mask)[0]
            if len(train_mask_indices) > 0:
                loss = torch.nn.functional.nll_loss(out[train_mask_indices], self.data.y[train_mask_indices])
                loss.backward()
                optimizer.step()
            else:
                print("Warning: No training nodes available")
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                out = self.model(self.data.x, self.data.edge_index, None)
                
                # Training accuracy
                pred = out.argmax(dim=1)
                train_mask_indices = torch.where(self.data.train_mask)[0]
                if len(train_mask_indices) > 0:
                    train_correct = pred[train_mask_indices] == self.data.y[train_mask_indices]
                    train_acc = int(train_correct.sum()) / len(train_mask_indices)
                else:
                    train_acc = 0.0
                
                # Validation accuracy
                val_mask_indices = torch.where(self.data.val_mask)[0]
                if len(val_mask_indices) > 0:
                    val_correct = pred[val_mask_indices] == self.data.y[val_mask_indices]
                    val_acc = int(val_correct.sum()) / len(val_mask_indices)
                else:
                    val_acc = 0.0
                
                # Print progress
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index, None)
            pred = out.argmax(dim=1)
            
            # Test accuracy
            test_mask_indices = torch.where(self.data.test_mask)[0]
            if len(test_mask_indices) > 0:
                test_correct = pred[test_mask_indices] == self.data.y[test_mask_indices]
                test_acc = int(test_correct.sum()) / len(test_mask_indices)
            else:
                test_acc = 0.0
            
            print(f"Final Test Accuracy: {test_acc:.4f}")
        
        # Save node embeddings for future use
        with torch.no_grad():
            self.node_embeddings = self.model.conv1(self.data.x, self.data.edge_index).cpu().numpy()
        
        # Return metrics
        metrics = {
            'test_accuracy': test_acc,
            'best_val_accuracy': best_val_acc
        }
        
        return metrics
    
def predict(self, technologies_list, top_k=3):
        """
        Predict job roles for a list of technologies.
        
        Args:
            technologies_list: List of technology strings
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predicted job roles and probabilities
        """
        if not self.model or not self.graph:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        # Find corresponding nodes for technologies
        tech_nodes = []
        for tech in technologies_list:
            if tech in self.tech_to_node:
                tech_nodes.append(self.tech_to_node[tech])
            else:
                # Try to find similar technology
                for known_tech, node_id in self.tech_to_node.items():
                    if tech.lower() in known_tech.lower() or known_tech.lower() in tech.lower():
                        tech_nodes.append(node_id)
                        break
        
        if not tech_nodes:
            # No matching technology found
            print("Warning: No matching technologies found in the graph")
            # Return uniform distribution over job roles
            uniform_prob = 1.0 / len(self.label_encoder.classes_)
            return {
                "predictions": [
                    {"role": role, "probability": uniform_prob} 
                    for role in self.label_encoder.classes_[:top_k]
                ]
            }
        
        # Create a subgraph with the selected technologies
        subgraph = nx.Graph()
        
        # Add technology nodes
        for node_id in tech_nodes:
            subgraph.add_node(
                node_id, 
                type='tech', 
                name=self.node_to_tech[node_id], 
                embedding=None
            )
        
        # Add job roles
        num_techs = len(self.tech_to_node)
        num_jobs = len(self.label_encoder.classes_)
        
        for job_id in range(num_jobs):
            node_id = num_techs + job_id
            subgraph.add_node(
                node_id, 
                type='job', 
                name=self.label_encoder.classes_[job_id], 
                job_id=job_id
            )
        
        # Add edges from the original graph
        for tech_node in tech_nodes:
            for job_node in range(num_techs, num_techs + num_jobs):
                if self.graph.has_edge(tech_node, job_node):
                    weight = self.graph[tech_node][job_node]['weight']
                    subgraph.add_edge(tech_node, job_node, weight=weight)
        
    # Convert subgraph to PyTorch Geometric Data
        data = self.prepare_pytorch_geometric_data(subgraph)
        data = data.to(DEVICE)
        
        # Run inference
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index, None)
            
            # Get probabilities for job nodes
            job_probs = torch.softmax(out[num_techs:num_techs + num_jobs], dim=1)
            
            # Sum probabilities over all classes to get job role scores
            job_scores = job_probs.sum(dim=1).cpu().numpy()
            
            # Normalize to get probabilities
            job_probs_normalized = job_scores / job_scores.sum()
            
            # Get top k predictions
            top_indices = np.argsort(job_probs_normalized)[::-1][:top_k]
            top_probabilities = job_probs_normalized[top_indices]
            
            # Convert indices to job role names
            top_roles = [self.label_encoder.classes_[idx] for idx in top_indices]
            
            # Format results
            predictions = {
                "predictions": [
                    {"role": role, "probability": float(prob)} 
                    for role, prob in zip(top_roles, top_probabilities)
                ]
            }
            
            return predictions
    
def save_model(self, output_dir=None):
        """
        Save the trained model and graph data.
        
        Args:
            output_dir: Directory to save the model in
        """
        if not self.model or not self.graph:
            raise ValueError("No trained model to save. Train a model first.")
        
        if output_dir is None:
            output_dir = os.path.join(MODEL_DIR, 'gnn_job_classifier')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save PyTorch model
        model_path = os.path.join(output_dir, 'gnn_model.pt')
        torch.save(self.model.state_dict(), model_path)
        
        # Save graph data and mappings
        graph_data = {
            'tech_to_node': self.tech_to_node,
            'node_to_tech': self.node_to_tech,
            'label_encoder': self.label_encoder,
            'node_embeddings': self.node_embeddings
        }
        
        graph_path = os.path.join(output_dir, 'graph_data.joblib')
        joblib.dump(graph_data, graph_path)
        
        # Save PyTorch Geometric data
        data_path = os.path.join(output_dir, 'pyg_data.pt')
        torch.save(self.data, data_path)
        
        # Save model configuration
        config = {
            'num_node_features': self.model.conv1.in_channels,
            'hidden_channels': self.model.conv1.out_channels,
            'num_classes': self.model.classifier[-1].out_features
        }
        
        config_path = os.path.join(output_dir, 'model_config.joblib')
        joblib.dump(config, config_path)
        
        print(f"Model and graph data saved to {output_dir}")
    
def load_model(self, model_dir):
        """
        Load a trained model and graph data.
        
        Args:
            model_dir: Directory containing the saved model
        """
        model_path = os.path.join(model_dir, 'gnn_model.pt')
        graph_path = os.path.join(model_dir, 'graph_data.joblib')
        data_path = os.path.join(model_dir, 'pyg_data.pt')
        config_path = os.path.join(model_dir, 'model_config.joblib')
        
        # Load graph data and mappings
        graph_data = joblib.load(graph_path)
        self.tech_to_node = graph_data['tech_to_node']
        self.node_to_tech = graph_data['node_to_tech']
        self.label_encoder = graph_data['label_encoder']
        self.node_embeddings = graph_data['node_embeddings']
        
        # Load PyTorch Geometric data
        self.data = torch.load(data_path, map_location=DEVICE)
        
        # Load model configuration
        config = joblib.load(config_path)
        
        # Initialize model with the same architecture
        self.model = JobClassifierGNN(
            num_node_features=config['num_node_features'],
            num_classes=config['num_classes'],
            hidden_channels=config['hidden_channels']
        ).to(DEVICE)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        
        # Reconstruct the graph (not saved directly due to potential serialization issues)
        # Note: We can reconstruct the graph if needed for visualization or analysis
        
        print(f"Model and graph data loaded from {model_dir}")


if __name__ == "__main__":
    # Example usage:
    data_path = os.path.join(DATA_PROCESSED_DIR, 'clean_v3.csv')
    
    classifier = GNNJobClassifier()
    classifier.train(data_path, hidden_channels=64, epochs=100)
    
    # Save the model
    classifier.save_model()
    
    # Example prediction
    technologies = ["Python", "JavaScript", "React", "MongoDB", "AWS"]
    predictions = classifier.predict(technologies)
    print("\nExample Prediction:")
    for pred in predictions["predictions"]:
        print(f"{pred['role']}: {pred['probability']:.4f}")
