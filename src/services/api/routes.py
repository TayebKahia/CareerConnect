import uuid
import time
import json
import os
from flask import jsonify, request
from typing import Dict, Any, Tuple

from src.config import DEFAULT_TOP_K, MAX_SEARCH_RESULTS, DEVICE
from src.utils.helpers import debug_log, process_user_input, enhance_job_details_with_onet
from src.utils.skill_processor import SkillProcessor
from models.classes.gnn_model_cache import gnn_model_cache

# Import necessary libraries for new model
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, Linear, LayerNorm
from sentence_transformers import SentenceTransformer
import numpy as np


# Define the HeteroGNN model class for the new endpoint
class HeteroGNN(nn.Module):
    def __init__(self, sbert_dim, hidden_channels, out_channels, num_layers, metadata, node_input_dims):
        super().__init__()
        self.out_channels = out_channels
        self.dropout = nn.Dropout(0.3)

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
                                                     concat=True, dropout=0.3, add_self_loops=(not is_bipartite))
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
            # Mode 1: Return all node embeddings
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


def register_routes(app, model_manager):
    # Initialize skill processor
    skill_processor = SkillProcessor()

    def initialize_new_model():
        # First try loading from the cache
        if gnn_model_cache.load_cache():
            return True

        # If cache loading fails, build from scratch
        try:
            # Path to model and data from config
            from src.config import NEW_MODEL_PATH, DATA_PATH
            model_path = NEW_MODEL_PATH
            data_path = DATA_PATH

            debug_log(
                f"Loading new model data from {model_path} and {data_path}...")

            # Load the data
            with open(data_path, 'r') as f:
                job_data_source = json.load(f)

            job_titles, tech_names_set, skill_category_names_set = [], set(), set()
            all_tech_demand_hotness_raw = []

            # Extract job titles, technologies, and skill categories
            for job_entry in job_data_source:
                if "title" not in job_entry:
                    continue
                job_titles.append(job_entry["title"])
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

            job_titles = sorted(list(set(job_titles)))
            tech_names = sorted(list(tech_names_set))
            skill_category_names = sorted(list(skill_category_names_set))

            # Create mappings
            job_map = {name: i for i, name in enumerate(job_titles)}
            tech_map = {name: i for i, name in enumerate(tech_names)}
            skill_category_map = {name: i for i,
                                  name in enumerate(skill_category_names)}
            job_id_to_title_map = {i: name for name, i in job_map.items()}
            tech_id_to_name_map = {i: name for name, i in tech_map.items()}

            # Load sentence transformer model
            sentence_model_name = "sentence-transformers/msmarco-distilbert-base-v4"
            sentence_model = SentenceTransformer(
                sentence_model_name, device=DEVICE)

            # Generate embeddings
            job_title_sbert_embeddings = sentence_model.encode(
                job_titles, convert_to_tensor=True, device=DEVICE)
            tech_name_sbert_embeddings = sentence_model.encode(
                tech_names, convert_to_tensor=True, device=DEVICE)
            skill_category_name_sbert_embeddings = sentence_model.encode(
                skill_category_names, convert_to_tensor=True, device=DEVICE)

            # Process tech features
            tech_name_to_raw_features = {item['name']: [
                item['demand'], item['hotness']] for item in all_tech_demand_hotness_raw}
            ordered_raw_features_for_scaling = [tech_name_to_raw_features.get(
                name, [0.0, 0.0]) for name in tech_names]

            if ordered_raw_features_for_scaling:
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                normalized_demand_hotness_ordered = scaler.fit_transform(
                    np.array(ordered_raw_features_for_scaling))
            else:
                normalized_demand_hotness_ordered = np.array([]).reshape(0, 2)

            sbert_dim = 768
            tech_features_list = []
            for i, name in enumerate(tech_names):
                name_emb = tech_name_sbert_embeddings[i].cpu()
                if normalized_demand_hotness_ordered.shape[0] > 0 and i < normalized_demand_hotness_ordered.shape[0]:
                    norm_demand = torch.tensor(
                        normalized_demand_hotness_ordered[i, 0], dtype=torch.float)
                    norm_hotness = torch.tensor(
                        normalized_demand_hotness_ordered[i, 1], dtype=torch.float)
                    features = torch.cat([name_emb, norm_demand.unsqueeze(
                        0), norm_hotness.unsqueeze(0)], dim=0)
                else:
                    features = torch.cat(
                        [name_emb, torch.tensor([0.0, 0.0])], dim=0)
                tech_features_list.append(features.to(DEVICE))

            tech_gnn_input_features = torch.stack(
                tech_features_list) if tech_features_list else torch.empty((0, sbert_dim + 2), device=DEVICE)

            # Create the graph data
            data = HeteroData()
            data['job'].x = job_title_sbert_embeddings
            data['technology'].x = tech_gnn_input_features
            data['skill_category'].x = skill_category_name_sbert_embeddings

            # Build graph edges
            edge_job_requires_category_src, edge_job_requires_category_dst = [], []
            edge_sc_contains_tech_src, edge_sc_contains_tech_dst = [], []
            edge_job_uses_tech_src, edge_job_uses_tech_dst = [], []
            job_to_true_tech_ids = {}

            for job_entry in job_data_source:
                if "title" not in job_entry or job_entry["title"] not in job_map:
                    continue

                job_id = job_map[job_entry["title"]]
                job_to_true_tech_ids[job_id] = set()

                if "technology_skills" in job_entry and job_entry["technology_skills"] is not None:
                    for sc_data in job_entry["technology_skills"]:
                        if "skill_title" not in sc_data or sc_data["skill_title"] not in skill_category_map:
                            continue

                        sc_id = skill_category_map[sc_data["skill_title"]]
                        edge_job_requires_category_src.append(job_id)
                        edge_job_requires_category_dst.append(sc_id)

                        if "technologies" in sc_data and sc_data["technologies"] is not None:
                            for tech_data in sc_data["technologies"]:
                                if "name" not in tech_data or tech_data["name"] not in tech_map:
                                    continue

                                tech_name = tech_data["name"]
                                tech_id = tech_map[tech_name]

                                edge_sc_contains_tech_src.append(sc_id)
                                edge_sc_contains_tech_dst.append(tech_id)
                                edge_job_uses_tech_src.append(job_id)
                                edge_job_uses_tech_dst.append(tech_id)
                                job_to_true_tech_ids[job_id].add(tech_id)

            data['job', 'requires_category', 'skill_category'].edge_index = torch.tensor(
                [edge_job_requires_category_src, edge_job_requires_category_dst], dtype=torch.long)
            data['skill_category', 'contains_tech', 'technology'].edge_index = torch.tensor(
                [edge_sc_contains_tech_src, edge_sc_contains_tech_dst], dtype=torch.long)
            data['job', 'uses_tech', 'technology'].edge_index = torch.tensor(
                [edge_job_uses_tech_src, edge_job_uses_tech_dst], dtype=torch.long) if edge_job_uses_tech_src else torch.empty((2, 0), dtype=torch.long)

            data = data.to(DEVICE)

            # Create and load the GNN model
            node_input_dims_map = {nt: data[nt].x.shape[1]
                                   for nt in data.node_types if data[nt].num_nodes > 0}
            for nt in data.metadata()[0]:
                if nt not in node_input_dims_map:
                    if nt == 'job':
                        node_input_dims_map[nt] = sbert_dim
                    elif nt == 'technology':
                        node_input_dims_map[nt] = sbert_dim + 2
                    elif nt == 'skill_category':
                        node_input_dims_map[nt] = sbert_dim
                    else:
                        debug_log(
                            f"Warning: Node type {nt} has no initial features and no default dim set.")

            # Load the model
            hidden_channels = 128
            gnn_out_channels = 128
            gnn_layers = 2
            model = HeteroGNN(
                sbert_dim=sbert_dim,
                hidden_channels=hidden_channels,
                out_channels=gnn_out_channels,
                num_layers=gnn_layers,
                metadata=data.metadata(),
                node_input_dims=node_input_dims_map
            )
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()

            # Store all data needed for prediction            # Store data in the cache
            gnn_model_cache.model = model
            gnn_model_cache.data = data
            gnn_model_cache.sentence_model = sentence_model
            gnn_model_cache.job_id_to_title_map = job_id_to_title_map
            gnn_model_cache.tech_map = tech_map
            gnn_model_cache.job_titles = job_titles

            # Save to cache for future use
            mappings = {
                'job_id_to_title_map': job_id_to_title_map,
                'tech_map': tech_map,
                'job_titles': job_titles
            }
            gnn_model_cache.save_cache(data, mappings)
            debug_log("New model initialized successfully")
            return True

        except Exception as e:
            debug_log(f"Error initializing new model: {str(e)}")
            return False

    @app.route('/', methods=['GET'])
    def index():
        """Root endpoint with API information"""
        return jsonify({
            'status': 'success',
            'message': 'Job Recommendation API is running',
            'available_endpoints': [
                {
                    'path': '/api/recommend-GNN-Onet',
                    'method': 'POST',
                    'description': 'Get job recommendations using the newer model (best_model1.pth)',
                    'example_payload': {
                        'skills': [
                            {'name': 'Python', 'type': 'technology',
                                'similarity': 1.0},
                            {'name': 'Machine learning',
                                'type': 'technology', 'similarity': 0.95},
                            {'name': 'Structured query language (SQL)',
                                'type': 'technology', 'similarity': 0.85}
                        ],
                        'top_n': 5
                    }
                },
                {
                    'path': '/api/recommend-GNN-Onet-from-text',
                    'method': 'POST',
                    'description': 'Get job recommendations from text description using the newer model (best_model1.pth)',
                    'example_payload': {
                        'text': 'I am a data scientist with expertise in Python, SQL, and machine learning...',
                        'top_n': 5
                    }
                },
                {
                    'path': '/api/skills',
                    'method': 'GET',
                    'description': 'Get available skills and technologies',
                },
                {
                    'path': '/api/skill-search',
                    'method': 'GET',
                    'description': 'Search for skills and technologies',
                    'parameters': {
                        'q': 'Search query',
                        'limit': 'Maximum number of results (default: 20)'
                    }
                }
            ]
        })

    @app.route('/api/recommend-GNN-Onet', methods=['POST'])
    def recommend_jobs_new():
        """Recommend jobs based on skills and technologies using the new model"""
        request_id = str(uuid.uuid4())[:8]
        debug_log(
            f"\n[{request_id}] ===== Starting New Job Recommendation Request (New Model) =====")

        try:
            # Initialize the model if not already initialized
            if not initialize_new_model():
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to initialize the new model'
                }), 500

            # Parse request data
            if request.content_type and 'application/json' in request.content_type:
                data = request.get_json(force=True, silent=True)
            else:
                try:
                    data = json.loads(request.data.decode('utf-8'))
                except:
                    data = None

            debug_log(
                f"[{request_id}] Received request data: {json.dumps(data, indent=2)}")

            if not data or 'skills' not in data:
                return jsonify({
                    'status': 'error',
                    'message': 'Missing required field: skills'
                }), 400            # Process input skills
            user_skills = []
            for skill in data['skills']:
                if 'name' not in skill:
                    continue

                name = skill['name']
                # Handle both input formats
                if 'similarity' in skill:
                    # Original format with similarity field
                    skill_type = skill.get('type', 'technology_name')
                    weight = float(skill.get('similarity', 1.0))
                else:
                    # New format with weight field
                    skill_type = skill.get('type', 'technology_name')
                    weight = float(skill.get('weight', 1.0))

                # Convert type to expected format for the model
                if skill_type == 'technology':
                    skill_type = 'technology_name'

                user_skills.append((name, skill_type, weight))

            top_n = int(data.get('top_n', DEFAULT_TOP_K))
            debug_log(
                f"[{request_id}] Processed {len(user_skills)} user skills for new model")

            if len(user_skills) == 0:
                return jsonify({
                    'status': 'error',
                    'message': 'No valid skills provided'
                }), 400

            # Predict jobs using the new model
            debug_log(f"[{request_id}] Running prediction with new model...")

            # Convert skills to format needed for model
            skill_names_for_embedding = [s[0] for s in user_skills]
            skill_weights = torch.tensor(
                [s[2] for s in user_skills], dtype=torch.float, device=DEVICE).unsqueeze(1)

            with torch.no_grad():                # Generate embeddings for skills using the cached model
                raw_query_sbert_embeddings_list = gnn_model_cache.sentence_model.encode(
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

                # Get job scores from model
                job_scores, _, _ = gnn_model_cache.model(
                    gnn_model_cache.data.x_dict,
                    gnn_model_cache.data.edge_index_dict,
                    aggregated_raw_sbert_q_user_emb
                )                # Sort jobs by scores
                ranked_scores, ranked_indices = torch.sort(
                    job_scores, descending=True)                # Format results

                # Get top-n scores for normalization
                top_n_scores = [ranked_scores[i].item()
                                for i in range(min(top_n, len(ranked_indices)))]

                # Apply sigmoid normalization with the same approach used in the models
                if top_n_scores:
                    scores_array = np.array(top_n_scores)
                    # Apply sigmoid normalization
                    normalized_scores = 1 / \
                        (1 + np.exp(-(scores_array - scores_array.mean()) /
                         (scores_array.std() + 1e-6)))
                    # Scale to 70-100 range
                    scaled_scores = 70 + 30 * (normalized_scores - normalized_scores.min()) / (
                        normalized_scores.max() - normalized_scores.min() + 1e-6)

                results = []
                for i in range(min(top_n, len(ranked_indices))):
                    pred_job_id = ranked_indices[i].item()
                    pred_job_title = gnn_model_cache.job_id_to_title_map.get(
                        pred_job_id, f"Unknown Job ID: {pred_job_id}")
                    raw_score = ranked_scores[i].item()

                    # Use the normalized scores instead of direct multiplication
                    scaled_score = scaled_scores[i] if top_n_scores else min(
                        100, max(0, raw_score * 100))
                    results.append({
                        "title": pred_job_title,
                        "raw_score": float(raw_score)
                    })

            debug_log(
                f"[{request_id}] ===== Completed Job Recommendation Request (New Model) =====\n")
            return jsonify({
                'status': 'success',
                'request_id': request_id,
                'recommendations': results,
                'total_recommendations': len(results),
                'timestamp': time.time()
            })

        except Exception as e:
            debug_log(
                f"[{request_id}] ERROR in new model recommendation: {str(e)}")
            debug_log(
                f"[{request_id}] ===== Failed Job Recommendation Request (New Model) =====\n")
            return jsonify({
                'status': 'error',
                'request_id': request_id,
                'message': f'Error processing request: {str(e)}'
            }), 500

    @app.route('/api/recommend-GNN-Onet-from-text', methods=['POST'])
    def recommend_from_text_new():
        """Recommend jobs based on text description using the new model"""
        request_id = str(uuid.uuid4())[:8]
        debug_log(
            f"\n[{request_id}] ===== Starting New Text-Based Recommendation Request (New Model) =====")

        try:
            # Initialize the model if not already initialized
            if not initialize_new_model():
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to initialize the new model'
                }), 500

            # Parse request data
            if request.content_type and 'application/json' in request.content_type:
                data = request.get_json(force=True, silent=True)
            else:
                try:
                    data = json.loads(request.data.decode('utf-8'))
                except:
                    data = None

            debug_log(
                f"[{request_id}] Received request data: {json.dumps(data, indent=2)}")

            if not data or 'text' not in data:
                return jsonify({
                    'status': 'error',
                    'message': 'Missing required field: text'
                }), 400

            # Process text and get recommendations
            text = data['text']
            top_n = int(data.get('top_n', DEFAULT_TOP_K))

            debug_log(
                f"[{request_id}] Input text length: {len(text)} characters")
            debug_log(f"[{request_id}] Text preview: {text[:200]}...")
            # Extract skills from text using the same skill processor as original
            debug_log(f"[{request_id}] Top-N value: {top_n}")
            debug_log(f"[{request_id}] Starting skill extraction...")
            filtered_skills = skill_processor.process_text(text)

            debug_log(
                f"\n[{request_id}] Extracted {len(filtered_skills)} skills:")
            for idx, skill in enumerate(filtered_skills, 1):
                debug_log(f"[{request_id}] {idx}. {skill['name']}")
                debug_log(f"[{request_id}]    Type: {skill['type']}")
                debug_log(
                    f"[{request_id}]    Similarity: {skill['similarity']:.3f}")

            # Load ONET job data
            onet_data_path = os.path.join(
                'data', 'processed', 'abbr_cleaned_IT_data_from_onet.json')
            with open(onet_data_path, 'r') as f:
                onet_data = json.load(f)

            # Create a lookup dictionary for O*NET data
            onet_data_dict = {
                job.get('title', ''): job for job in onet_data if 'title' in job}

            # Convert extracted skills to the format expected by the new model
            user_skills = []
            extracted_skill_names = set()
            for skill in filtered_skills:
                if skill['similarity'] >= 0.5:  # Using the same threshold as original
                    name = skill['name']
                    skill_type = 'technology_name'  # Always use technology_name for consistency
                    weight = float(skill['similarity'])
                    user_skills.append((name, skill_type, weight))
                    extracted_skill_names.add(name.lower())

            debug_log(
                f"[{request_id}] Processed {len(user_skills)} user skills for new model")

            if len(user_skills) == 0:
                return jsonify({
                    'status': 'error',
                    'message': 'No valid skills extracted from text'
                }), 400

            # Predict jobs using the new model
            debug_log(f"[{request_id}] Running prediction with new model...")

            # Convert skills to format needed for model
            skill_names_for_embedding = [s[0] for s in user_skills]
            skill_weights = torch.tensor(
                [s[2] for s in user_skills], dtype=torch.float, device=DEVICE).unsqueeze(1)

            with torch.no_grad():
                # Generate embeddings for skills using the cached model
                raw_query_sbert_embeddings_list = gnn_model_cache.sentence_model.encode(
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

                # Get job scores from model
                job_scores, _, _ = gnn_model_cache.model(
                    gnn_model_cache.data.x_dict,
                    gnn_model_cache.data.edge_index_dict,
                    aggregated_raw_sbert_q_user_emb
                )                # Sort jobs by scores
                ranked_scores, ranked_indices = torch.sort(
                    job_scores, descending=True)

                # Format results with full job data
                results = []
                for i in range(min(top_n, len(ranked_indices))):
                    pred_job_id = ranked_indices[i].item()
                    pred_job_title = gnn_model_cache.job_id_to_title_map.get(
                        pred_job_id, f"Unknown Job ID: {pred_job_id}")
                    raw_score = ranked_scores[i].item()

                    # Get full job data from ONET
                    job_data = onet_data_dict.get(pred_job_title, {})

                    # Add matching flags for skills and technologies
                    if 'technology_skills' in job_data:
                        for tech_skill in job_data['technology_skills']:
                            # Add matching flag for skill category
                            tech_skill['is_skill_matched'] = tech_skill['skill_title'].lower(
                            ) in extracted_skill_names

                            # Add matching flags for technologies
                            if 'technologies' in tech_skill:
                                for tech in tech_skill['technologies']:
                                    tech['is_matched'] = tech['name'].lower(
                                    ) in extracted_skill_names

                    # Combine all data
                    result = {
                        "title": pred_job_title,
                        "raw_score": float(raw_score),
                        "job_data": job_data  # Full O*NET data for the job
                    }
                    results.append(result)

            response = {
                'status': 'success',
                'request_id': request_id,
                'recommendations': results,
                'total_recommendations': len(results),
                'extracted_skills': filtered_skills,
                'total_skills': len(filtered_skills),
                'timestamp': time.time()
            }

            debug_log(
                f"[{request_id}] ===== Completed Text-Based Recommendation Request (New Model) =====\n")
            return jsonify(response)

        except Exception as e:
            debug_log(
                f"[{request_id}] ERROR in text-based recommendation (New Model): {str(e)}")
            debug_log(
                f"[{request_id}] ===== Failed Text-Based Recommendation Request (New Model) =====\n")
            return jsonify({
                'status': 'error',
                'request_id': request_id,
                'message': f'Error processing request: {str(e)}'}), 500

    @app.route('/api/skills', methods=['GET'])
    def get_available_skills():
        """Get the list of available skills and technologies"""
        try:
            skills = sorted(list(model_manager.system.unique_skills))
            technologies = sorted(
                list(model_manager.system.unique_technologies))

            return jsonify({
                'status': 'success',
                'skills': skills,
                'technologies': technologies,
                'total_skills': len(skills),
                'total_technologies': len(technologies)
            })

        except Exception as e:
            debug_log(f"Error getting skills: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Error processing request: {str(e)}'
            }), 500

    @app.route('/api/skill-search', methods=['GET'])
    def search_skills():
        """Search for skills and technologies by name prefix"""
        try:
            query = request.args.get('q', '').lower()
            limit = min(int(request.args.get(
                'limit', MAX_SEARCH_RESULTS)), MAX_SEARCH_RESULTS)

            if not query:
                return jsonify({
                    'status': 'error',
                    'message': 'Missing required query parameter: q'
                }), 400

            matching_skills = [
                skill for skill in model_manager.system.unique_skills
                if query in skill.lower()
            ]
            matching_techs = [
                tech for tech in model_manager.system.unique_technologies
                if query in tech.lower()
            ]

            matching_skills = matching_skills[:limit]
            matching_techs = matching_techs[:limit]

            return jsonify({
                'status': 'success',
                'skills': matching_skills,
                'technologies': matching_techs,
                'total_matches': len(matching_skills) + len(matching_techs)
            })

        except Exception as e:
            debug_log(f"Error searching skills: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Error processing request: {str(e)}'
            }), 500
