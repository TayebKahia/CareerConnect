# Temporary file to store the new endpoint
import os
import json
from flask import Flask, jsonify, request
import torch
import numpy as np


def create_endpoint(app):
    @app.route('/api/recommend-from-text-new', methods=['POST'])
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
            debug_log(f"[{request_id}] Top-N value: {top_n}")

            # Extract skills from text using the same skill processor as original endpoint
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
                )

                # Sort jobs by scores
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

            debug_log(f"\n[{request_id}] Response summary:")
            debug_log(
                f"[{request_id}] - Total recommendations: {len(results)}")
            debug_log(
                f"[{request_id}] - Total extracted skills: {len(filtered_skills)}")
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
