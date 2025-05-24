"""
Traditional ML-based job classifier using XGBoost and feature engineering.
This serves as one component of the Hybrid Ensemble model.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2
import xgboost as xgb
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

class MLJobClassifier:
    """
    Traditional ML-based job classifier using XGBoost.
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
        self.feature_selector = None  # Feature selector for dimensionality reduction
        
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
    
    def select_features(self, X, y, k=1000):
        """
        Select the most important features using chi-squared test.
        
        Args:
            X: Feature matrix
            y: Target labels
            k: Number of features to select
            
        Returns:
            X_selected: Reduced feature matrix
        """
        # Use selectKBest to choose the top k features
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(chi2, k=min(k, X.shape[1]))
            self.feature_selector.fit(X, y)
        
        return self.feature_selector.transform(X)
    
    def train(self, data_path, feature_selection=True, n_features=1000):
        """
        Train the XGBoost classifier on the dataset.
        
        Args:
            data_path: Path to the CSV file containing the dataset
            feature_selection: Whether to apply feature selection
            n_features: Number of features to select if feature_selection is True
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Load and preprocess data
        data = pd.read_csv(data_path)
        X, y = self.preprocess_data(data)
        
        # Apply feature selection if enabled
        if feature_selection and X.shape[1] > n_features:
            X = self.select_features(X, y, k=n_features)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define XGBoost parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': len(self.label_encoder.classes_),
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'eval_metric': 'mlogloss',
            'seed': 42
        }
        
        # Train XGBoost model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        self.model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=200, 
            evals=watchlist,
            early_stopping_rounds=20,
            verbose_eval=10
        )
        
        # Evaluate model
        y_pred = self.model.predict(dtest)
        y_pred_labels = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_labels)
        macro_f1 = f1_score(y_test, y_pred_labels, average='macro')
        weighted_f1 = f1_score(y_test, y_pred_labels, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test, 
            y_pred_labels, 
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
        
        # Apply feature selection if it was used during training
        if self.feature_selector:
            X = self.feature_selector.transform(X)
        
        # Make prediction
        dmatrix = xgb.DMatrix(X)
        probabilities = self.model.predict(dmatrix)[0]
        
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
            output_dir = os.path.join(MODEL_DIR, 'ml_job_classifier')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save XGBoost model
        model_path = os.path.join(output_dir, 'xgboost_model.json')
        self.model.save_model(model_path)
        
        # Save preprocessing components
        components = {
            'mlb': self.mlb,
            'label_encoder': self.label_encoder,
            'feature_selector': self.feature_selector
        }
        
        components_path = os.path.join(output_dir, 'preprocessing_components.joblib')
        joblib.dump(components, components_path)
        
        print(f"Model and components saved to {output_dir}")
    
    def load_model(self, model_dir):
        """
        Load a trained model and preprocessing components.
        
        Args:
            model_dir: Directory containing the saved model
        """
        model_path = os.path.join(model_dir, 'xgboost_model.json')
        components_path = os.path.join(model_dir, 'preprocessing_components.joblib')
        
        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        # Load preprocessing components
        components = joblib.load(components_path)
        self.mlb = components['mlb']
        self.label_encoder = components['label_encoder']
        self.feature_selector = components['feature_selector']
        
        print(f"Model and components loaded from {model_dir}")


if __name__ == "__main__":
    # Example usage:
    data_path = os.path.join(DATA_PROCESSED_DIR, 'clean_v3.csv')
    
    classifier = MLJobClassifier()
    classifier.train(data_path, feature_selection=True, n_features=500)
    
    # Save the model
    classifier.save_model()
    
    # Example prediction
    technologies = ["Python", "JavaScript", "React", "MongoDB", "AWS"]
    predictions = classifier.predict(technologies)
    print("\nExample Prediction:")
    for pred in predictions["predictions"]:
        print(f"{pred['role']}: {pred['probability']:.4f}")
