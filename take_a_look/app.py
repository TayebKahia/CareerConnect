import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, Linear, LayerNorm
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Dict
import uvicorn
import os
from contextlib import asynccontextmanager

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

SENTENCE_MODEL_NAME = 'sentence-transformers/msmarco-distilbert-base-v4'
SBERT_DIM = 768
HIDDEN_CHANNELS = 128
GNN_OUT_CHANNELS = 128
GNN_LAYERS = 2
DROPOUT = 0.3

# Path to model and data
MODEL_PATH = 'best_model1.pth'
DATA_PATH = 'filtered_IT_data.json'


class HeteroGNN(nn.Module):
    def __init__(self, sbert_dim, hidden_channels, out_channels, num_layers, metadata, node_input_dims):
        super().__init__()
        self.out_channels = out_channels
        self.dropout = nn.Dropout(DROPOUT)

        self.node_lins_initial = nn.ModuleDict()
        self.node_layer_norms_initial = nn.ModuleDict()
        for node_type in metadata[0]:
            self.node_lins_initial[node_type] = Linear(
                node_input_dims[node_type], hidden_channels)
            self.node_layer_norms_initial[node_type] = LayerNorm(
                hidden_channels)

        self.convs = nn.ModuleList()
        self.inter_lins_gnn = nn.ModuleList()

        for i in range(num_layers):
            conv_dict = {}
            for src, rel, dst in metadata[1]:
                is_bipartite = src != dst
                conv_dict[(src, rel, dst)] = GATConv((-1, -1), hidden_channels, heads=2,
                                                     concat=True, dropout=DROPOUT, add_self_loops=(not is_bipartite))
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

            layer_inter_projection = nn.ModuleDict()
            for node_type in metadata[0]:
                layer_inter_projection[node_type] = Linear(
                    hidden_channels * 2, hidden_channels)
            self.inter_lins_gnn.append(layer_inter_projection)

        self.job_final_projection = Linear(hidden_channels, out_channels)
        self.query_final_projection = Linear(sbert_dim, out_channels)

    def forward(self, x_dict_input, edge_index_dict, aggregated_query_sbert_embedding=None):
        """
        Forward function that can operate in two modes:
        1. If aggregated_query_sbert_embedding is None: compute node embeddings for the entire graph
        2. If aggregated_query_sbert_embedding is provided: compute job recommendation scores
        """
        if aggregated_query_sbert_embedding is None:
            # Mode 1: Return all node embeddings (for compatibility with existing code)
            return self._compute_node_embeddings_dict(x_dict_input, edge_index_dict)
        else:
            # Mode 2: Return job scores for the given query embedding
            all_node_embeddings = self._compute_node_embeddings_dict(
                x_dict_input, edge_index_dict)
            all_job_embeddings = all_node_embeddings['job']

            # Project query embedding to the same space as job embeddings
            projected_query_emb = self.project_query_embedding(
                aggregated_query_sbert_embedding)

            # Compute job scores directly (dot product)
            job_scores = torch.matmul(
                all_job_embeddings, projected_query_emb.unsqueeze(-1)).squeeze(-1)

            return job_scores, all_job_embeddings, projected_query_emb

    def _compute_node_embeddings_dict(self, x_dict_input, edge_index_dict):
        """Compute embeddings for all nodes in the graph"""
        x_dict = {}
        for node_type, x_in in x_dict_input.items():
            x = self.node_lins_initial[node_type](x_in)
            x = self.node_layer_norms_initial[node_type](x)
            x = self.dropout(F.relu(x))
            x_dict[node_type] = x

        for i, conv_layer in enumerate(self.convs):
            x_dict_updates = conv_layer(x_dict, edge_index_dict)
            for node_type, x_gat_output in x_dict_updates.items():
                x_projected = self.inter_lins_gnn[i][node_type](x_gat_output)
                x_dict[node_type] = self.dropout(F.relu(x_projected))

        if 'job' in x_dict:
            x_dict['job'] = self.job_final_projection(x_dict['job'])
        return x_dict

    def project_query_embedding(self, raw_aggregated_sbert_embedding):
        """Project a raw SBERT embedding to the job embedding space"""
        return self.query_final_projection(raw_aggregated_sbert_embedding)


class Skill(BaseModel):
    name: str
    type: str = "technology_name"
    weight: float = 1.0


class JobRecommendationRequest(BaseModel):
    skills: List[Skill]
    top_n: int = 3


class JobRecommendation(BaseModel):
    title: str
    score: float


class JobRecommendationResponse(BaseModel):
    recommendations: List[JobRecommendation]

# Create a context manager to load the model and data


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the data
    print(f"Loading job data from {DATA_PATH}...")
    with open(DATA_PATH, 'r') as f:
        job_data_source = json.load(f)

    app.state.job_titles, tech_names_set, skill_category_names_set = [], set(), set()
    all_tech_demand_hotness_raw = []

    # Extract job titles, technologies, and skill categories
    for job_entry in job_data_source:
        if "title" not in job_entry:
            continue
        app.state.job_titles.append(job_entry["title"])
        if "technology_skills" in job_entry and job_entry["technology_skills"] is not None:
            for sc_data in job_entry["technology_skills"]:
                if "skill_title" not in sc_data:
                    continue
                skill_category_names_set.add(sc_data["skill_title"])
                if "technologies" in sc_data and sc_data["technologies"] is not None:
                    for tech_data in sc_data["technologies"]:
                        if "name" not in tech_data:
                            continue
                        tech_names_set.add(tech_data["name"])
                        try:
                            demand = float(
                                str(tech_data.get("demand_percentage", "0")).replace('%', ''))
                            hotness = float(
                                str(tech_data.get("hot_tech_percentage", "0")).replace('%', ''))
                            all_tech_demand_hotness_raw.append(
                                {'name': tech_data["name"], 'demand': demand, 'hotness': hotness})
                        except ValueError:
                            all_tech_demand_hotness_raw.append(
                                {'name': tech_data["name"], 'demand': 0.0, 'hotness': 0.0})

    app.state.job_titles = sorted(list(set(app.state.job_titles)))
    app.state.tech_names = sorted(list(tech_names_set))
    app.state.skill_category_names = sorted(list(skill_category_names_set))

    # Create mappings between names and IDs
    app.state.job_map = {name: i for i,
                         name in enumerate(app.state.job_titles)}
    app.state.tech_map = {name: i for i,
                          name in enumerate(app.state.tech_names)}
    app.state.skill_category_map = {
        name: i for i, name in enumerate(app.state.skill_category_names)}

    # Create reverse mappings
    app.state.job_id_to_title_map = {
        i: name for name, i in app.state.job_map.items()}
    app.state.tech_id_to_name_map = {
        i: name for name, i in app.state.tech_map.items()}

    print(f"Loaded {len(app.state.job_titles)} jobs, {len(app.state.tech_names)} technologies, and {len(app.state.skill_category_names)} skill categories.")

    # Load sentence transformer model
    print("Loading sentence transformer model...")
    app.state.sentence_model = SentenceTransformer(
        SENTENCE_MODEL_NAME, device=DEVICE)

    # Generate embeddings
    app.state.job_title_sbert_embeddings = app.state.sentence_model.encode(
        app.state.job_titles, convert_to_tensor=True, device=DEVICE)
    app.state.tech_name_sbert_embeddings = app.state.sentence_model.encode(
        app.state.tech_names, convert_to_tensor=True, device=DEVICE)
    app.state.skill_category_name_sbert_embeddings = app.state.sentence_model.encode(
        app.state.skill_category_names, convert_to_tensor=True, device=DEVICE)

    # Process tech features
    tech_name_to_raw_features = {item['name']: [
        item['demand'], item['hotness']] for item in all_tech_demand_hotness_raw}
    ordered_raw_features_for_scaling = [tech_name_to_raw_features.get(
        name, [0.0, 0.0]) for name in app.state.tech_names]

    if ordered_raw_features_for_scaling:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized_demand_hotness_ordered = scaler.fit_transform(
            np.array(ordered_raw_features_for_scaling))
    else:
        normalized_demand_hotness_ordered = np.array([]).reshape(0, 2)

    tech_features_list = []
    for i, name in enumerate(app.state.tech_names):
        name_emb = app.state.tech_name_sbert_embeddings[i].cpu()
        if normalized_demand_hotness_ordered.shape[0] > 0 and i < normalized_demand_hotness_ordered.shape[0]:
            norm_demand = torch.tensor(
                normalized_demand_hotness_ordered[i, 0], dtype=torch.float)
            norm_hotness = torch.tensor(
                normalized_demand_hotness_ordered[i, 1], dtype=torch.float)
            features = torch.cat([name_emb, norm_demand.unsqueeze(
                0), norm_hotness.unsqueeze(0)], dim=0)
        else:
            features = torch.cat([name_emb, torch.tensor([0.0, 0.0])], dim=0)
        tech_features_list.append(features.to(DEVICE))

    app.state.tech_gnn_input_features = torch.stack(
        tech_features_list) if tech_features_list else torch.empty((0, SBERT_DIM + 2), device=DEVICE)

    # Create the graph data
    app.state.data = HeteroData()
    app.state.data['job'].x = app.state.job_title_sbert_embeddings
    app.state.data['technology'].x = app.state.tech_gnn_input_features
    app.state.data['skill_category'].x = app.state.skill_category_name_sbert_embeddings

    # Build graph edges
    edge_job_requires_category_src, edge_job_requires_category_dst = [], []
    edge_sc_contains_tech_src, edge_sc_contains_tech_dst = [], []
    edge_job_uses_tech_src, edge_job_uses_tech_dst = [], []
    app.state.job_to_true_tech_ids = {}

    for job_entry in job_data_source:
        if "title" not in job_entry or job_entry["title"] not in app.state.job_map:
            continue

        job_id = app.state.job_map[job_entry["title"]]
        app.state.job_to_true_tech_ids[job_id] = set()

        if "technology_skills" in job_entry and job_entry["technology_skills"] is not None:
            for sc_data in job_entry["technology_skills"]:
                if "skill_title" not in sc_data or sc_data["skill_title"] not in app.state.skill_category_map:
                    continue

                sc_id = app.state.skill_category_map[sc_data["skill_title"]]
                edge_job_requires_category_src.append(job_id)
                edge_job_requires_category_dst.append(sc_id)

                if "technologies" in sc_data and sc_data["technologies"] is not None:
                    for tech_data in sc_data["technologies"]:
                        if "name" not in tech_data or tech_data["name"] not in app.state.tech_map:
                            continue

                        tech_name = tech_data["name"]
                        tech_id = app.state.tech_map[tech_name]

                        edge_sc_contains_tech_src.append(sc_id)
                        edge_sc_contains_tech_dst.append(tech_id)
                        edge_job_uses_tech_src.append(job_id)
                        edge_job_uses_tech_dst.append(tech_id)
                        app.state.job_to_true_tech_ids[job_id].add(tech_id)

    app.state.data['job', 'requires_category', 'skill_category'].edge_index = torch.tensor(
        [edge_job_requires_category_src, edge_job_requires_category_dst], dtype=torch.long)
    app.state.data['skill_category', 'contains_tech', 'technology'].edge_index = torch.tensor(
        [edge_sc_contains_tech_src, edge_sc_contains_tech_dst], dtype=torch.long)
    app.state.data['job', 'uses_tech', 'technology'].edge_index = torch.tensor(
        [edge_job_uses_tech_src, edge_job_uses_tech_dst], dtype=torch.long) if edge_job_uses_tech_src else torch.empty((2, 0), dtype=torch.long)

    app.state.data = app.state.data.to(DEVICE)

    # Create and load GNN model
    node_input_dims_map = {nt: app.state.data[nt].x.shape[1]
                           for nt in app.state.data.node_types if app.state.data[nt].num_nodes > 0}
    for nt in app.state.data.metadata()[0]:
        if nt not in node_input_dims_map:
            if nt == 'job':
                node_input_dims_map[nt] = SBERT_DIM
            elif nt == 'technology':
                node_input_dims_map[nt] = SBERT_DIM + 2
            elif nt == 'skill_category':
                node_input_dims_map[nt] = SBERT_DIM
            else:
                print(
                    f"Warning: Node type {nt} has no initial features and no default dim set.")

    print(f"Loading model from {MODEL_PATH}...")
    app.state.model = HeteroGNN(
        sbert_dim=SBERT_DIM, hidden_channels=HIDDEN_CHANNELS, out_channels=GNN_OUT_CHANNELS,
        num_layers=GNN_LAYERS, metadata=app.state.data.metadata(), node_input_dims=node_input_dims_map
    )
    app.state.model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE))
    app.state.model.to(DEVICE)
    app.state.model.eval()
    print("Model loaded successfully.")

    yield

    # Clean up resources
    print("Cleaning up resources...")
    # No explicit cleanup needed for PyTorch models

# Initialize FastAPI app
app = FastAPI(
    title="Job Recommendation API",
    description="An API that recommends jobs based on user skills using a Graph Neural Network",
    version="1.0.0",
    lifespan=lifespan
)


def predict_jobs(user_skills: List[Skill], model, data, sentence_model, job_id_title_map, top_n=3):
    """
    Predict job recommendations based on user skills.
    """
    with torch.no_grad():
        if not user_skills:
            return []

        # Convert skills to format needed for model
        skill_tuples = [(skill.name, skill.type, skill.weight)
                        for skill in user_skills]
        skill_names_for_embedding = [s.name for s in user_skills]
        skill_weights = torch.tensor(
            [s.weight for s in user_skills], dtype=torch.float, device=DEVICE).unsqueeze(1)

        if not skill_names_for_embedding:
            return []

        # Generate embeddings for the skills
        raw_query_sbert_embeddings_list = sentence_model.encode(
            skill_names_for_embedding, convert_to_tensor=True, device=DEVICE)

        # Aggregate embeddings with weights
        if raw_query_sbert_embeddings_list.ndim == 1 and len(skill_names_for_embedding) == 1:
            aggregated_raw_sbert_q_user_emb = raw_query_sbert_embeddings_list
        elif raw_query_sbert_embeddings_list.ndim > 1 and raw_query_sbert_embeddings_list.size(0) > 0:
            if skill_weights.numel() == raw_query_sbert_embeddings_list.size(0):
                weighted_embs = raw_query_sbert_embeddings_list * skill_weights
                aggregated_raw_sbert_q_user_emb = torch.sum(
                    weighted_embs, dim=0) / (torch.sum(skill_weights) + 1e-9)
            else:
                aggregated_raw_sbert_q_user_emb = torch.mean(
                    raw_query_sbert_embeddings_list, dim=0)
        else:
            return []

        if aggregated_raw_sbert_q_user_emb.ndim == 0 or aggregated_raw_sbert_q_user_emb.numel() == 0:
            return []

        # Get job scores from model
        job_scores, _, _ = model(
            data.x_dict, data.edge_index_dict, aggregated_raw_sbert_q_user_emb)

        # Sort jobs by scores
        ranked_scores, ranked_indices = torch.sort(job_scores, descending=True)

        results = []
        for i in range(min(top_n, len(ranked_indices))):
            pred_job_id = ranked_indices[i].item()
            pred_job_title = job_id_title_map.get(
                pred_job_id, f"Unknown Job ID: {pred_job_id}")
            score = ranked_scores[i].item()
            results.append({"title": pred_job_title, "score": float(score)})
        return results


@app.get("/")
def read_root():
    """
    Root endpoint returning basic API information
    """
    return {
        "name": "Job Recommendation API",
        "description": "API for recommending jobs based on user skills",
        "version": "1.0.0"
    }


@app.post("/recommend", response_model=JobRecommendationResponse)
def recommend_jobs(request: JobRecommendationRequest):
    """
    Recommend jobs based on provided skills
    """
    if not request.skills:
        raise HTTPException(
            status_code=400, detail="No skills provided in request")

    recommendations = predict_jobs(
        request.skills,
        app.state.model,
        app.state.data,
        app.state.sentence_model,
        app.state.job_id_to_title_map,
        top_n=request.top_n
    )

    if not recommendations:
        raise HTTPException(
            status_code=404, detail="Could not generate recommendations with the provided skills")

    return JobRecommendationResponse(recommendations=recommendations)


@app.get("/skills/technologies", response_model=List[str])
def get_technologies():
    """
    Get a list of all available technologies
    """
    return app.state.tech_names


@app.get("/skills/categories", response_model=List[str])
def get_skill_categories():
    """
    Get a list of all available skill categories
    """
    return app.state.skill_category_names


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
