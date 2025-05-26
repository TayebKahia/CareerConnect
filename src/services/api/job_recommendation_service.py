import os
import time
import json
import uuid
import torch
import numpy as np
import traceback
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
import random

from src.config import DEVICE, DEFAULT_TOP_K, NEW_MODEL_PATH, DATA_PATH
from src.utils.helpers import debug_log
from src.utils.skill_processor import SkillProcessor
from models.mb.classes.gnn_model_cache import gnn_model_cache
from models.mb.classes.hetero_gnn_recommendation import HeteroGNN
import pypdfium2 as pdfium  # For PDF processing


class JobRecommendationService:
    def __init__(self):
        self.skill_processor = SkillProcessor()
        self._initialized = False
        self._sentence_model = None

    def initialize_new_model(self):
        """Initialize the new GNN model, either from cache or from scratch"""
        # Check if model is already initialized
        if gnn_model_cache.model is not None:
            return True

        # First try loading from the cache
        if gnn_model_cache.load_cache():
            return True

        # If cache loading fails, build from scratch
        try:
            # Path to model and data from config
            model_path = NEW_MODEL_PATH
            data_path = DATA_PATH

            debug_log(
                f"Loading new model data from {model_path} and {data_path}...")

            # Track timing for profiling
            start_time = time.time()
            debug_log("Loading job data...")

            # Load the data
            with open(data_path, 'r') as f:
                job_data_source = json.load(f)

            debug_log(
                f"Loaded job data in {time.time() - start_time:.2f} seconds")

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
            debug_log("Loading sentence transformer model...")
            st_start_time = time.time()
            sentence_model = SentenceTransformer(
                sentence_model_name, device=DEVICE)
            debug_log(
                f"Loaded sentence transformer in {time.time() - st_start_time:.2f} seconds")

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

            # Store all data needed for prediction
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

    def recommend_jobs_from_skills(self, data, request_id):
        """Get job recommendations based on provided skills"""
        debug_log(
            f"\n[{request_id}] ===== Starting New Job Recommendation Request (New Model) =====")

        try:
            # The model should already be initialized at startup
            # Only try to initialize if something went wrong (e.g., server restart)
            if gnn_model_cache.model is None and not self.initialize_new_model():
                return {
                    'status': 'error',
                    'message': 'Failed to initialize the new model'
                }, 500

            debug_log(
                f"[{request_id}] Received request data: {json.dumps(data, indent=2)}")

            if not data or 'skills' not in data:
                return {
                    'status': 'error',
                    'message': 'Missing required field: skills'
                }, 400

            # Process input skills
            user_skills = []
            for skill in data['skills']:
                if 'name' not in skill:
                    continue

                name = skill['name']
                # Handle both input formats
                if 'similarity' in skill:
                    skill_type = skill.get('type', 'technology_name')
                    weight = float(skill.get('similarity', 1.0))
                else:
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
                return {
                    'status': 'error',
                    'message': 'No valid skills provided'
                }, 400

            # Predict jobs using the new model
            debug_log(f"[{request_id}] Running prediction with new model...")
            results = self._get_recommendations(user_skills, top_n)

            debug_log(
                f"[{request_id}] ===== Completed Job Recommendation Request (New Model) =====\n")
            return {
                'status': 'success',
                'request_id': request_id,
                'recommendations': results,
                'total_recommendations': len(results),
                'timestamp': time.time()
            }

        except Exception as e:
            debug_log(
                f"[{request_id}] ERROR in new model recommendation: {str(e)}")
            debug_log(
                f"[{request_id}] ===== Failed Job Recommendation Request (New Model) =====\n")
            return {
                'status': 'error',
                'request_id': request_id,
                'message': f'Error processing request: {str(e)}'
            }, 500

    def recommend_from_text(self, data, request_id):
        """Get job recommendations from text description"""
        debug_log(
            f"\n[{request_id}] ===== Starting New Text-Based Recommendation Request (New Model) =====")

        try:
            # Initialize the model if not already initialized
            if not self.initialize_new_model():
                return {
                    'status': 'error',
                    'message': 'Failed to initialize the new model'
                }, 500

            debug_log(
                f"[{request_id}] Received request data: {json.dumps(data, indent=2)}")

            if not data or 'text' not in data:
                return {
                    'status': 'error',
                    'message': 'Missing required field: text'
                }, 400

            # Process text and get recommendations
            text = data['text']
            top_n = int(data.get('top_n', DEFAULT_TOP_K))

            debug_log(
                f"[{request_id}] Input text length: {len(text)} characters")
            debug_log(f"[{request_id}] Text preview: {text[:200]}...")
            debug_log(f"[{request_id}] Top-N value: {top_n}")
            debug_log(f"[{request_id}] Starting skill extraction...")

            # Extract skills from text using skill processor
            filtered_skills = self.skill_processor.process_text(text)

            debug_log(
                f"\n[{request_id}] Extracted {len(filtered_skills)} skills:")
            for idx, skill in enumerate(filtered_skills, 1):
                debug_log(f"[{request_id}] {idx}. {skill['name']}")
                debug_log(f"[{request_id}]    Type: {skill['type']}")
                debug_log(
                    f"[{request_id}]    Similarity: {skill['similarity']:.3f}")

            # Load ONET job data
            onet_data = self._load_onet_data(request_id)

            # Convert extracted skills to the format expected by the model
            user_skills = []
            extracted_skill_names = set()
            for skill in filtered_skills:
                if skill['similarity'] >= 0.8:  # Using threshold for high-confidence matches
                    name = skill['name']
                    skill_type = 'technology_name'  # Consistent format
                    weight = float(skill['similarity'])
                    user_skills.append((name, skill_type, weight))
                    extracted_skill_names.add(name.lower())

            debug_log(
                f"[{request_id}] Processed {len(user_skills)} user skills for new model")

            if len(user_skills) == 0:
                return {
                    'status': 'error',
                    'message': 'No valid skills extracted from text'
                }, 400

            # Predict jobs using the model
            debug_log(f"[{request_id}] Running prediction with new model...")
            results = self._get_recommendations(user_skills, top_n)

            # Enhance results with ONET data
            enhanced_results = self._enhance_results_with_onet(
                results, onet_data, extracted_skill_names, filtered_skills)

            debug_log(
                f"[{request_id}] ===== Completed Text-Based Recommendation Request (New Model) =====\n")
            return {
                'status': 'success',
                'request_id': request_id,
                'recommendations': enhanced_results,
                'total_recommendations': len(enhanced_results),
                'extracted_skills': filtered_skills,
                'total_skills': len(filtered_skills),
                'timestamp': time.time()
            }

        except Exception as e:
            debug_log(
                f"[{request_id}] ERROR in text-based recommendation: {str(e)}")
            debug_log(
                f"[{request_id}] ===== Failed Text-Based Recommendation Request (New Model) =====\n")
            return {
                'status': 'error',
                'request_id': request_id,
                'message': f'Error processing request: {str(e)}'
            }, 500

    def recommend_from_cv(self, file, top_n, request_id):
        """Get job recommendations from uploaded CV (PDF)"""
        debug_log(
            f"\n[{request_id}] ===== Starting CV-Based Recommendation Request (New Model) =====")

        try:
            # Initialize the model if not already initialized
            if not self.initialize_new_model():
                return {
                    'status': 'error',
                    'message': 'Failed to initialize the new model'
                }, 500

            debug_log(
                f"[{request_id}] Processing uploaded CV: {file.filename}")

            # Extract text from PDF using pypdfium2
            try:
                pdf_doc = pdfium.PdfDocument(file)
                text_pages = []
                for i in range(len(pdf_doc)):
                    page = pdf_doc.get_page(i)
                    textpage = page.get_textpage()
                    text_pages.append(textpage.get_text_range())
                    textpage.close()
                    page.close()
                text = "\n\n".join(text_pages)
                pdf_doc.close()

                debug_log(
                    f"[{request_id}] Successfully extracted {len(text)} characters from PDF using pypdfium2")

                # Save extracted text for debugging
                debug_output_dir = os.path.join('debug_output')
                os.makedirs(debug_output_dir, exist_ok=True)

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                safe_filename = ''.join(
                    c if c.isalnum() else '_' for c in file.filename)
                output_filename = f"{safe_filename}_{timestamp}_{request_id}.txt"
                output_filepath = os.path.join(
                    debug_output_dir, output_filename)

                with open(output_filepath, 'w', encoding='utf-8') as f:
                    f.write(
                        f"=== Extracted Text from {file.filename} (using pypdfium2) ===\n\n")
                    f.write(text)

                debug_log(
                    f"[{request_id}] Saved extracted text to {output_filepath}")

                if not text.strip():
                    return {
                        'status': 'error',
                        'message': 'No text could be extracted from the PDF'
                    }, 400

            except Exception as e:
                debug_log(
                    f"[{request_id}] Error extracting text from PDF: {str(e)}")
                return {
                    'status': 'error',
                    'message': f'Error processing PDF: {str(e)}'
                }, 400

            debug_log(f"[{request_id}] CV text preview: {text[:500]}...")
            debug_log(f"[{request_id}] Top-N value: {top_n}")
            debug_log(f"[{request_id}] Starting skill extraction...")

            # Extract skills from text using skill processor
            filtered_skills = self.skill_processor.process_text(text)

            debug_log(
                f"\n[{request_id}] Extracted {len(filtered_skills)} skills from CV:")
            for idx, skill in enumerate(filtered_skills, 1):
                debug_log(f"[{request_id}] {idx}. {skill['name']}")
                debug_log(f"[{request_id}]    Type: {skill['type']}")
                debug_log(
                    f"[{request_id}]    Similarity: {skill['similarity']:.3f}")

            # Load ONET job data
            onet_data = self._load_onet_data(request_id)

            # Convert extracted skills to the format expected by the model
            user_skills = []
            extracted_skill_names = set()
            for skill in filtered_skills:
                if skill['similarity'] >= 0.8:  # Using threshold for high-confidence matches
                    name = skill['name']
                    skill_type = 'technology_name'  # Consistent format
                    weight = float(skill['similarity'])
                    user_skills.append((name, skill_type, weight))
                    extracted_skill_names.add(name.lower())

            debug_log(
                f"[{request_id}] Processed {len(user_skills)} user skills for new model")

            if len(user_skills) == 0:
                return {
                    'status': 'error',
                    'message': 'No valid skills extracted from CV'
                }, 400

            # Predict jobs using the model
            debug_log(f"[{request_id}] Running prediction with new model...")
            results = self._get_recommendations(user_skills, top_n)

            # Enhance results with ONET data
            enhanced_results = self._enhance_results_with_onet(
                results, onet_data, extracted_skill_names, filtered_skills)

            debug_log(
                f"[{request_id}] ===== Completed CV-Based Recommendation Request (New Model) =====\n")
            return {
                'status': 'success',
                'request_id': request_id,
                'recommendations': enhanced_results,
                'total_recommendations': len(enhanced_results),
                'extracted_skills': filtered_skills,
                'total_skills': len(filtered_skills),
                'timestamp': time.time()
            }

        except Exception as e:
            debug_log(
                f"[{request_id}] ERROR in CV-based recommendation: {str(e)}")
            debug_log(
                f"[{request_id}] ===== Failed CV-Based Recommendation Request (New Model) =====\n")
            return {
                'status': 'error',
                'request_id': request_id,
                'message': f'Error processing request: {str(e)}'
            }, 500

    def _get_recommendations(self, user_skills, top_n):
        """Get job recommendations using the GNN model"""
        with torch.no_grad():
            # Convert skills to format needed for model
            skill_names_for_embedding = [s[0] for s in user_skills]
            skill_weights = torch.tensor(
                [s[2] for s in user_skills], dtype=torch.float, device=DEVICE).unsqueeze(1)

            # Generate and aggregate embeddings
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
            )

            # Sort jobs by scores
            ranked_scores, ranked_indices = torch.sort(
                job_scores, descending=True)

            # Format results
            top_n_scores = [ranked_scores[i].item()
                            for i in range(min(top_n, len(ranked_indices)))]

            # Apply sigmoid normalization
            if top_n_scores:
                scores_array = np.array(top_n_scores)
                normalized_scores = 1 / \
                    (1 + np.exp(-(scores_array - scores_array.mean()) /
                     (scores_array.std() + 1e-6)))
                scaled_scores = 70 + 30 * (normalized_scores - normalized_scores.min()) / (
                    normalized_scores.max() - normalized_scores.min() + 1e-6)

            results = []
            for i in range(min(top_n, len(ranked_indices))):
                pred_job_id = ranked_indices[i].item()
                pred_job_title = gnn_model_cache.job_id_to_title_map.get(
                    pred_job_id, f"Unknown Job ID: {pred_job_id}")
                matchScore = ranked_scores[i].item()

                # Use normalized score if available
                scaled_score = scaled_scores[i] if top_n_scores else min(
                    100, max(0, matchScore * 100))

                results.append({
                    "title": pred_job_title,
                    "matchScore": float(matchScore),
                    "score": float(scaled_score)
                })

            return results

    def _load_onet_data(self, request_id):
        """Load O*NET job data from file"""
        onet_data_path = os.path.join(
            'data', 'processed', 'abbr_cleaned_IT_data_from_onet.json')
        try:
            with open(onet_data_path, 'r') as f:
                try:
                    onet_data = json.load(f)
                except json.JSONDecodeError as json_error:
                    debug_log(
                        f"[{request_id}] JSON parse error with O*NET data: {str(json_error)}")
                    debug_log(
                        f"[{request_id}] Error occurred near line {json_error.lineno}, column {json_error.colno}")
                    # Use a fallback approach - load filtered data instead
                    fallback_path = os.path.join(
                        'data', 'processed', 'filtered_IT_data.json')
                    debug_log(
                        f"[{request_id}] Attempting to load fallback data from: {fallback_path}")
                    with open(fallback_path, 'r') as fallback_file:
                        onet_data = json.load(fallback_file)

            debug_log(
                f"[{request_id}] Successfully loaded {len(onet_data)} ONET job records")
            return onet_data

        except Exception as e:
            debug_log(f"[{request_id}] Error loading O*NET data: {str(e)}")
            debug_log(f"[{request_id}] {traceback.format_exc()}")
            return []  # Return empty list as fallback

    def _enhance_results_with_onet(self, results, onet_data, extracted_skill_names, filtered_skills):
        """Enhance recommendation results with ONET job data"""
        # Create a lookup dictionary for O*NET data
        onet_data_dict = {
            job.get('title', ''): job for job in onet_data if 'title' in job
        }

        # Extract original matchScores
        match_scores = [result['matchScore'] for result in results]
        # Avoid division by zero
        max_score = max(match_scores) if match_scores else 1

        enhanced_results = []
        for result in results:
            job_title = result['title']
            job_data = onet_data_dict.get(job_title, {})

            # Add matching flags for skills and technologies
            if 'technology_skills' in job_data:
                for tech_skill in job_data['technology_skills']:
                    tech_skill['is_skill_matched'] = tech_skill['skill_title'].lower(
                    ) in extracted_skill_names

                    if 'technologies' in tech_skill:
                        for tech in tech_skill['technologies']:
                            tech['is_matched'] = tech['name'].lower(
                            ) in extracted_skill_names

            # Normalize and scale matchScore (to a max of ~80)
            raw_score = result['matchScore']
            # allows slight variation around 80
            scale_cap = random.uniform(78, 84)
            normalized_score = (raw_score / max_score) * scale_cap

            # Get salary from job_data or use a default fallback
            salary = job_data.get("salary", 100000)

            enhanced_result = {
                "title": job_title,
                "matchScore": round(normalized_score, 2),
                "keySkills": filtered_skills,
                "salary": f"{salary}$",  # optional formatting as string
                # "job_data": job_data
            }

            enhanced_results.append(enhanced_result)

        return enhanced_results
