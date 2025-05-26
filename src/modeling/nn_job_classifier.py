"""
Neural network-based job classifier using technology embeddings.
This serves as one component of the Hybrid Ensemble model.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Define project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


class TechDataset(Dataset):
    """Dataset for technology features and job labels."""
    
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class JobClassifierNN(nn.Module):
    """Neural network architecture for job classification."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5):
        super(JobClassifierNN, self).__init__()
        
        # Define layers with dynamic hidden dimensions
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        
        # Output layer
        self.output = nn.Linear(hidden_dims[-1], output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Pass through hidden layers with ReLU and dropout
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        # Output layer (no activation; will use cross-entropy loss)
        x = self.output(x)
        return x


class NNJobClassifier:
    """
    Neural network-based job classifier using technology embeddings.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to load a pre-trained model from.
        """
        self.model = None
        self.mlb = None  # MultiLabelBinarizer for technology encoding
        self.label_encoder = None  # LabelEncoder for job roles
        
        if model_path:
            self.load_model(model_path)
    
    def preprocess_data(self, data):
        """
        Preprocess the data for training or prediction.
        
        Args:
            data: DataFrame containing DevType and technology columns
            
        Returns:
            X: Feature matrix
            y: Target labels (if DevType is present)
        """        # Convert semicolon-separated technology strings to lists (handle both str and list)
        for col in TECH_COLUMNS:
            if col in data.columns:
                data[col] = data[col].fillna('').apply(
                    lambda x: x if isinstance(x, list) else ([t.strip() for t in x.split(';')] if x else [])
                )
        
        # If this is the first time processing, fit the MultiLabelBinarizer
        if self.mlb is None:
            self.mlb = {}
            for col in TECH_COLUMNS:
                if col in data.columns:
                    self.mlb[col] = MultiLabelBinarizer()
                    self.mlb[col].fit(data[col])
        
        # Transform each technology column and concatenate
        X_parts = []
        for col in TECH_COLUMNS:
            if col in data.columns and col in self.mlb:
                X_part = self.mlb[col].transform(data[col])
                X_parts.append(X_part)
        
        # Concatenate all technology features
        X = np.concatenate(X_parts, axis=1) if X_parts else np.array([])
        
        # If DevType is present, encode the labels
        if 'DevType' in data.columns:
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(data['DevType'])
            
            y = self.label_encoder.transform(data['DevType'])
            return X, y
        
        return X
    
    def train(self, data_path, hidden_dims=[512, 256, 128], dropout=0.5, 
              batch_size=64, epochs=20, learning_rate=0.001):
        """
        Train the neural network classifier on the dataset.
        
        Args:
            data_path: Path to the CSV file containing the dataset
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability for regularization
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Load and preprocess data
        data = pd.read_csv(data_path)
        X, y = self.preprocess_data(data)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create datasets and dataloaders
        train_dataset = TechDataset(X_train, y_train)
        test_dataset = TechDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model
        input_dim = X.shape[1]
        output_dim = len(self.label_encoder.classes_)
        
        self.model = JobClassifierNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        ).to(DEVICE)
          # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    
                    # Get predictions
                    _, predicted = torch.max(outputs, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            avg_val_loss = val_loss / len(test_loader)
            scheduler.step(avg_val_loss)
            
            # Calculate metrics
            val_accuracy = accuracy_score(val_targets, val_preds)
            val_macro_f1 = f1_score(val_targets, val_preds, average='macro')
            
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f}, "
                  f"Val Macro F1: {val_macro_f1:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict().copy()
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        self.model.eval()
        y_pred = []
        
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(DEVICE)
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.cpu().numpy())
        
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1
        }
        
        print("\nFinal Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test, 
            y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
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
        if not self.model:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        # Create a dataframe with the technologies
        data = pd.DataFrame({
            col: [[t for t in technologies_list if t in self.mlb[col].classes_]] 
            if col in self.mlb else [[]] 
            for col in TECH_COLUMNS
        }, index=[0])
        
        # Preprocess the data
        X = self.preprocess_data(data)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Get top k predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_probabilities = probabilities[top_indices]
        
        # Convert indices back to job role names
        top_roles = self.label_encoder.inverse_transform(top_indices)
        
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
        Save the trained model and preprocessing components.
        
        Args:
            output_dir: Directory to save the model in
        """
        if not self.model:
            raise ValueError("No trained model to save. Train a model first.")
        
        if output_dir is None:
            output_dir = os.path.join(MODEL_DIR, 'nn_job_classifier')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save PyTorch model
        model_path = os.path.join(output_dir, 'nn_model.pt')
        torch.save(self.model.state_dict(), model_path)
        
        # Save model architecture (input_dim, hidden_dims, output_dim)
        model_config = {
            'input_dim': self.model.layers[0].in_features,
            'hidden_dims': [layer.out_features for layer in self.model.layers],
            'output_dim': self.model.output.out_features
        }
        
        # Save preprocessing components and model config
        components = {
            'mlb': self.mlb,
            'label_encoder': self.label_encoder,
            'model_config': model_config
        }
        
        components_path = os.path.join(output_dir, 'nn_components.joblib')
        joblib.dump(components, components_path)
        
        print(f"Model and components saved to {output_dir}")
    
    def load_model(self, model_dir):
        """
        Load a trained model and preprocessing components.
        
        Args:
            model_dir: Directory containing the saved model
        """
        model_path = os.path.join(model_dir, 'nn_model.pt')
        components_path = os.path.join(model_dir, 'nn_components.joblib')
        
        # Load preprocessing components and model config
        components = joblib.load(components_path)
        self.mlb = components['mlb']
        self.label_encoder = components['label_encoder']
        model_config = components['model_config']
        
        # Initialize model with the same architecture
        self.model = JobClassifierNN(
            input_dim=model_config['input_dim'],
            hidden_dims=model_config['hidden_dims'],
            output_dim=model_config['output_dim']
        ).to(DEVICE)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        
        print(f"Model and components loaded from {model_dir}")


if __name__ == "__main__":
    # Example usage:
    data_path = os.path.join(DATA_PROCESSED_DIR, 'clean_v3.csv')
    
    classifier = NNJobClassifier()
    classifier.train(
        data_path,
        hidden_dims=[512, 256, 128],
        dropout=0.5,
        batch_size=64,
        epochs=20
    )
    
    # Save the model
    classifier.save_model()
    
    # Example prediction
    technologies = ["Python", "JavaScript", "React", "MongoDB", "AWS"]
    predictions = classifier.predict(technologies)
    print("\nExample Prediction:")
    for pred in predictions["predictions"]:
        print(f"{pred['role']}: {pred['probability']:.4f}")
