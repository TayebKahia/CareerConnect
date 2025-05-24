# Job Role Prediction - Ensemble Models

This module provides ensemble approaches for predicting job roles based on technology skills. We offer two different ensemble implementations:

1. **Hybrid Ensemble Model** - Combines ML, NN, and GNN for maximum accuracy
2. **Dual Ensemble Model** - Combines only ML and NN for faster inference with good accuracy

## Architecture Overview

### Hybrid Ensemble Model

The Hybrid Ensemble model combines three complementary approaches:

1. **Traditional ML Classifier (XGBoost)** - A gradient boosting model that excels with tabular data
2. **Neural Network Classifier** - A deep learning approach that captures complex relationships between technologies
3. **Graph Neural Network (GNN)** - A graph-based approach that leverages the relationships between technologies and job roles

### Dual Ensemble Model (New!)

The Dual Ensemble model combines two complementary approaches for fast and efficient prediction:

1. **Traditional ML Classifier (XGBoost)** - A gradient boosting model that excels with tabular data
2. **Neural Network Classifier** - A deep learning approach that captures complex relationships between technologies

![Architecture Diagram](docs/images/hybrid_ensemble_architecture.png)

## Key Components

- **ConceptMatcher**: Extracts technology concepts from text input
- **Hybrid Ensemble Classifier**: Combines predictions from multiple models
- **Job Prediction API**: FastAPI endpoints for prediction services

## Getting Started

### Installation

The model requires the dependencies listed in requirements.txt:

```bash
pip install -r requirements.txt
```

### Training the Models

#### Training the Hybrid Ensemble (ML, NN, GNN)

To train the full hybrid ensemble model:

```bash
cd src/modeling
python train_ensemble.py --data ../../data/processed/clean_v3.csv
```

Options:
- `--data`: Path to training data
- `--output`: Custom output directory
- `--weights`: Custom model weights (3 float values for ML, NN, GNN)
- `--eval-only`: Only evaluate without training

#### Training the Dual Ensemble (ML, NN only)

To train the dual ensemble model (faster and more resource-efficient):

```bash
cd src/modeling
python train_dual_ensemble.py --data ../../data/processed/clean_v3.csv
```

Options:
- `--data`: Path to training data
- `--output`: Custom output directory
- `--weights`: Custom model weights (2 float values for ML, NN)
- `--eval-only`: Only evaluate without training

### Using the Models

#### Command Line Usage

##### Hybrid Ensemble (ML, NN, GNN)

```bash
python predict_job_role.py --text "I am a software developer with experience in Python, JavaScript, and React."
```

Options:
- `--text`: Input text to analyze
- `--tech-file`: File with technologies (one per line)
- `--top-k`: Number of top predictions (default: 3)
- `--detailed`: Show detailed component predictions

##### Dual Ensemble (ML, NN only)

```bash
python predict_job_role_dual.py --text "I am a software developer with experience in Python, JavaScript, and React."
```

Options:
- `--text`: Input text to analyze
- `--tech-file`: File with technologies (one per line)
- `--top-k`: Number of top predictions (default: 3)
- `--detailed`: Show detailed component predictions

#### Python API

##### Hybrid Ensemble (ML, NN, GNN)

```python
from src.modeling.predict import predict_job_role

# Predict from text
result = predict_job_role(text="I have experience in Python, JavaScript, and React")

# Predict from technologies list
technologies = ["Python", "JavaScript", "React", "MongoDB", "AWS"]
result = predict_job_role(technologies=technologies)

# Get predictions
for pred in result["ensemble_predictions"]:
    print(f"{pred['role']}: {pred['probability']:.4f}")
```

##### Dual Ensemble (ML, NN only)

```python
from src.modeling.predict_dual import predict_job_role

# Predict from text
result = predict_job_role(text="I have experience in Python, JavaScript, and React")

# Predict from technologies list
technologies = ["Python", "JavaScript", "React", "MongoDB", "AWS"]
result = predict_job_role(technologies=technologies)

# Get predictions
for pred in result["ensemble_predictions"]:
    print(f"{pred['role']}: {pred['probability']:.4f}")
```

#### REST API

The model is also exposed through FastAPI endpoints:

```bash
cd src/services
uvicorn api.job_prediction_api:app --reload
```

API Documentation: http://localhost:8000/docs

## Model Evaluation

### Hybrid Ensemble Performance

The Hybrid Ensemble model was evaluated on a test dataset and achieved:

- Accuracy: 0.82
- Macro F1 Score: 0.79
- Weighted F1 Score: 0.81

### Dual Ensemble Performance

The Dual Ensemble model achieves competitive performance while using fewer resources:

- Accuracy: 0.80
- Macro F1 Score: 0.77
- Weighted F1 Score: 0.79

Both ensemble approaches consistently outperform their individual component models.

## Implementation Notes

### Class Structure

- **HybridEnsembleClassifier**: Core ensemble that combines ML, NN, and GNN models
- **DualEnsembleClassifier**: Lightweight ensemble that combines only ML and NN models
- **MLJobClassifier**: Traditional ML implementation using XGBoost
- **NNJobClassifier**: Neural network implementation 
- **GNNJobClassifier**: Graph neural network implementation (used only in Hybrid Ensemble)

### Weight Optimization

The ensemble weights are automatically optimized based on validation performance of each component model.

### Performance Comparison

| Model Type | Accuracy | Inference Speed | Resource Usage |
|------------|----------|----------------|----------------|
| Hybrid Ensemble | 0.82 | Moderate | High |
| Dual Ensemble | 0.80 | Fast | Moderate |

## Contributors

This implementation was created by the CareerConnect team.
