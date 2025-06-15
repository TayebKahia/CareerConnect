# New Job Recommendation API Endpoints

This document describes the new job recommendation endpoints that use the improved GNN model trained with PyTorch Geometric.

## 1. Skills-based Recommendations

### 1.1. Endpoint Details

- **URL**: `/api/recommend-GNN-Onet`
- **Method**: `POST`
- **Content Type**: `application/json`

### 1.2. Request Format

The endpoint accepts JSON data with the following structure:

```json
{
  "skills": [
    {
      "name": "Python",
      "type": "technology",
      "similarity": 1.0
    },
    {
      "name": "Machine learning",
      "type": "technology",
      "similarity": 0.95
    },
    {
      "name": "Structured query language (SQL)",
      "type": "technology",
      "similarity": 0.85
    }
  ],
  "top_n": 5
}
```

### 1.3. Request Parameters

- `skills`: Array of skill objects
  - `name`: The name of the technology or skill
  - `type`: The type of skill, usually "technology" or "skill"
  - `similarity`: The relevance/importance of this skill (between 0.0 and 1.0)
- `top_n`: Number of job recommendations to return (default is 5)

### 1.4. Alternative Format

The endpoint also supports the following alternative format:

```json
{
  "skills": [
    {
      "name": "Python",
      "type": "technology_name",
      "weight": 1.0
    },
    {
      "name": "Machine learning",
      "type": "technology_name",
      "weight": 0.95
    }
  ],
  "top_n": 5
}
```

With this format, use `weight` instead of `similarity` and `technology_name` instead of `technology`.

### 1.5. Response Format

The endpoint returns JSON data with the following structure:

```json
{
  "status": "success",
  "request_id": "abc123de",
  "recommendations": [
    {
      "title": "Data Scientist",
      "score": 87.5,
      "raw_score": 0.875
    },
    {
      "title": "Machine Learning Engineer",
      "score": 82.3,
      "raw_score": 0.823
    }
  ],
  "total_recommendations": 2,
  "timestamp": 1684761234.56
}
```

### 1.6. Response Parameters

- `status`: Success or error status
- `request_id`: Unique identifier for this request
- `recommendations`: Array of job recommendation objects
  - `title`: The job title
  - `score`: The confidence score (0-100)
  - `raw_score`: The raw model prediction score (0-1)
- `total_recommendations`: Total number of recommendations returned
- `timestamp`: Unix timestamp of the request

## 2. Text-based Recommendations

### 2.1. Endpoint Details

- **URL**: `/api/recommend-GNN-Onet-from-text`
- **Method**: `POST`
- **Content Type**: `application/json`

### 2.2. Request Format

The endpoint accepts JSON data with the following structure:

```json
{
  "text": "I am a data scientist with expertise in Python and machine learning. I have experience with TensorFlow, PyTorch, and SQL databases.",
  "top_n": 5
}
```

### 2.3. Request Parameters

- `text`: Free-form text describing skills, experience, and background
- `top_n`: Number of job recommendations to return (default is 5)

### 2.4. Response Format

The endpoint returns JSON data with the following structure:

```json
{
  "status": "success",
  "request_id": "xyz789ab",
  "recommendations": [
    {
      "title": "Data Scientist",
      "raw_score": 0.875,
      "job_data": {
        "title": "Data Scientist",
        "technology_skills": [
          {
            "skill_title": "Machine Learning",
            "is_skill_matched": true,
            "technologies": [
              {
                "name": "TensorFlow",
                "is_matched": true
              },
              {
                "name": "PyTorch",
                "is_matched": true
              }
            ]
          }
        ]
      }
    }
  ],
  "total_recommendations": 1,
  "extracted_skills": [
    {
      "name": "Python",
      "type": "technology_name",
      "similarity": 1.0
    },
    {
      "name": "Machine Learning",
      "type": "technology_name",
      "similarity": 0.95
    }
  ],
  "total_skills": 2,
  "timestamp": 1684761234.56
}
```

### 2.5. Response Parameters

- `status`: Success or error status
- `request_id`: Unique identifier for this request
- `recommendations`: Array of job recommendation objects
  - `title`: The job title
  - `raw_score`: The raw model prediction score (0-1)
  - `job_data`: Full O\*NET data for the job, including:
    - Technology skills with matching flags
    - Matched technologies within each skill
- `total_recommendations`: Total number of recommendations returned
- `extracted_skills`: Array of skills extracted from the input text
  - `name`: The name of the extracted skill
  - `type`: Type of the skill (usually "technology_name")
  - `similarity`: Confidence score for the extracted skill (0-1)
- `total_skills`: Total number of skills extracted
- `timestamp`: Unix timestamp of the request

## 3. Technical Implementation

The job recommendation endpoints use a heterogeneous graph neural network (HGNN) model saved at `models/best_model1.pth`. The model architecture includes:

### 3.1. Data Processing

- Converts input skills into embeddings using SentenceTransformer (msmarco-distilbert-base-v4)
- Normalizes skill weights/similarity scores
- Maps skills to known graph nodes

### 3.2. Model Architecture

- Heterogeneous graph with job, skill, and technology nodes
- Node features enriched with demand and hotness scores
- Multi-head graph attention layers for message passing
- Edge features incorporating TF-IDF weights
- Auxiliary prediction tasks for demand and hotness

### 3.3. Recommendation Process

- Aggregates user skill embeddings with attention weights
- Propagates information through the graph using HGT layers
- Computes job similarity scores using learned representations
- Applies auxiliary predictions to refine rankings
- Returns top-N jobs with confidence scores
