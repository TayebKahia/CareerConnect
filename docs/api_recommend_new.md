# New Job Recommendation API Endpoint

This document describes the new `/api/recommend-new` endpoint that uses the improved GNN model trained with PyTorch Geometric.

## Endpoint Details

- **URL**: `/api/recommend-new`
- **Method**: `POST`
- **Content Type**: `application/json`

## Request Parameters

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

### Parameters Explanation:

- `skills`: Array of skill objects
  - `name`: The name of the technology or skill
  - `type`: The type of skill, usually "technology" or "skill"
  - `similarity`: The relevance/importance of this skill (between 0.0 and 1.0)
- `top_n`: Number of job recommendations to return (default is 5)

#### Alternative Format (Also Supported)

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

## Response Format

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

### Response Fields:

- `status`: Success or error status
- `request_id`: Unique identifier for this request
- `recommendations`: Array of job recommendation objects
  - `title`: The job title
  - `score`: The confidence score (0-100)
  - `raw_score`: The raw model prediction score
- `total_recommendations`: The number of recommendations returned
- `timestamp`: Unix timestamp of the response

## Error Handling

In case of errors, the endpoint returns:

```json
{
  "status": "error",
  "request_id": "abc123de",
  "message": "Error message description"
}
```

## Implementation Details

This endpoint uses the GNN model saved at `models/best_model1.pth`. The model:

1. Accepts user skills as input
2. Computes embeddings using a sentence transformer
3. Aggregates embeddings based on weights
4. Uses a heterogeneous graph neural network to predict job recommendations
5. Returns ranked jobs with confidence scores

The model was trained on IT job data and can recommend jobs based on a user's skills and technologies.
