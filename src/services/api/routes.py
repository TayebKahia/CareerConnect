import uuid
import time
import json
from flask import jsonify, request
from typing import Dict, Any, Tuple

from src.config import DEFAULT_TOP_K, MAX_SEARCH_RESULTS
from src.utils.helpers import debug_log, process_user_input, enhance_job_details_with_onet
from src.utils.skill_processor import SkillProcessor
from src.modeling.models.hetero_gnn_recommendation import predict_job_titles_hetero


def register_routes(app, model_manager):
    # Initialize skill processor
    skill_processor = SkillProcessor()

    @app.route('/', methods=['GET'])
    def index():
        """Root endpoint with API information"""
        return jsonify({
            'status': 'success',
            'message': 'Job Recommendation API is running',
            'available_endpoints': [
                {
                    'path': '/api/recommend',
                    'method': 'POST',
                    'description': 'Get job recommendations based on skills and technologies',
                    'example_payload': {
                        'skills': [
                            {'name': 'Python', 'type': 'technology',
                                'similarity': 1.0},
                            {'name': 'Machine learning',
                                'type': 'skill', 'similarity': 0.95},
                            {'name': 'Data analysis',
                                'type': 'skill', 'similarity': 1.0}
                        ],
                        'top_k': 5
                    }
                },
                {
                    'path': '/api/recommend-from-text',
                    'method': 'POST',
                    'description': 'Get job recommendations from text description',
                    'example_payload': {
                        'text': 'I am a data scientist with expertise in Python, SQL, and machine learning...',
                        'top_k': 5
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

    @app.route('/api/recommend', methods=['POST'])
    def recommend_jobs():
        """Recommend jobs based on user skills and technologies"""
        request_id = str(uuid.uuid4())[:8]
        debug_log(
            f"\n[{request_id}] ===== Starting New Job Recommendation Request =====")

        try:
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
                }), 400

            # Process request
            normalized_skills = [
                {
                    'name': skill['name'],
                    'type': 'technology_name',  # Always use technology_name for consistency
                    'similarity': skill.get('similarity', 1.0)
                }
                for skill in data['skills']
            ]
            user_skills = process_user_input(normalized_skills)
            top_k = int(data.get('top_k', DEFAULT_TOP_K))

            debug_log(
                f"[{request_id}] Processed {len(user_skills)} user skills")
            debug_log(f"[{request_id}] Top-K value: {top_k}")
            for skill in user_skills:
                name, type_, score = skill
                debug_log(
                    f"[{request_id}] - Skill: {name} | Type: {type_} | Score: {score:.2f}")

            if len(user_skills) == 0:
                return jsonify({
                    'status': 'error',
                    'message': 'No valid skills provided'
                }), 400

            debug_log(f"[{request_id}] Calling job prediction model...")
            # Get job recommendations
            top_jobs, job_details = predict_job_titles_hetero(
                model_manager.model,
                model_manager.hetero_data,
                model_manager.system,
                user_skills,
                top_k=top_k
            )

            debug_log(
                f"[{request_id}] Received {len(top_jobs)} job predictions")

            # Format results
            results = []
            for job_title, confidence in top_jobs:
                debug_log(
                    f"[{request_id}] Processing job: {job_title} (confidence: {confidence:.3f})")
                details = job_details[job_title]
                enhanced_job = enhance_job_details_with_onet(
                    job_title,
                    details,
                    user_skills,
                    model_manager.original_job_data,
                    model_manager.onet_job_mapping
                )

                if enhanced_job:
                    enhanced_job['confidence'] = float(confidence)
                    results.append(enhanced_job)
                    debug_log(
                        f"[{request_id}] - Enhanced job details: {len(enhanced_job.get('required_skills', []))} required skills, {len(enhanced_job.get('matching_skills', []))} matching skills")

            debug_log(
                f"[{request_id}] ===== Completed Job Recommendation Request =====\n")
            return jsonify({
                'status': 'success',
                'request_id': request_id,
                'recommendations': results,
                'total_recommendations': len(results),
                'timestamp': time.time()
            })

        except Exception as e:
            debug_log(f"[{request_id}] ERROR in recommendation: {str(e)}")
            debug_log(
                f"[{request_id}] ===== Failed Job Recommendation Request =====\n")
            return jsonify({
                'status': 'error',
                'request_id': request_id,
                'message': f'Error processing request: {str(e)}'
            }), 500

    @app.route('/api/recommend-from-text', methods=['POST'])
    def recommend_from_text():
        """Recommend jobs based on text description"""
        request_id = str(uuid.uuid4())[:8]
        debug_log(
            f"\n[{request_id}] ===== Starting New Text-Based Recommendation Request =====")

        try:
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
            top_k = int(data.get('top_k', DEFAULT_TOP_K))

            debug_log(
                f"[{request_id}] Input text length: {len(text)} characters")
            debug_log(f"[{request_id}] Text preview: {text[:200]}...")
            debug_log(f"[{request_id}] Top-K value: {top_k}")

            # Extract skills from text
            debug_log(f"[{request_id}] Starting skill extraction...")
            filtered_skills = skill_processor.process_text(text)

            debug_log(
                f"\n[{request_id}] Extracted {len(filtered_skills)} skills:")
            for idx, skill in enumerate(filtered_skills, 1):
                debug_log(f"[{request_id}] {idx}. {skill['name']}")
                debug_log(f"[{request_id}]    Type: {skill['type']}")
                debug_log(
                    f"[{request_id}]    Similarity: {skill['similarity']:.3f}")

            # Format skills for recommendation
            skills_for_recommendation = [
                {
                    'name': skill['name'],
                    'type': 'technology_name',  # Always use technology_name for consistency
                    'similarity': skill['similarity']
                }
                for skill in filtered_skills
                # Match the threshold used in /recommend
                if skill['similarity'] >= 0.5
            ]

            debug_log(f"\n[{request_id}] Formatted skills for recommendation:")
            for skill in skills_for_recommendation:
                debug_log(
                    f"[{request_id}] - {skill['name']} ({skill['type']}): {skill['similarity']:.3f}")

            # Process request using the same helper as /recommend
            user_skills = process_user_input(skills_for_recommendation)
            debug_log(
                f"[{request_id}] Processed {len(user_skills)} skills through process_user_input")

            debug_log(f"\n[{request_id}] Calling job prediction model...")
            # Get recommendations using the same logic as /api/recommend
            top_jobs, job_details = predict_job_titles_hetero(
                model_manager.model,
                model_manager.hetero_data,
                model_manager.system,
                user_skills,  # Use processed skills
                top_k=top_k
            )

            debug_log(
                f"[{request_id}] Received {len(top_jobs)} job predictions")

            # Format results
            results = []
            debug_log(f"\n[{request_id}] Processing job recommendations:")
            for job_title, confidence in top_jobs:
                debug_log(f"[{request_id}] Processing job: {job_title}")
                debug_log(f"[{request_id}] - Confidence: {confidence:.3f}")

                details = job_details[job_title]
                enhanced_job = enhance_job_details_with_onet(
                    job_title,
                    details,
                    user_skills,  # Use processed skills here too, just like in /recommend
                    model_manager.original_job_data,
                    model_manager.onet_job_mapping
                )

                if enhanced_job:
                    enhanced_job['confidence'] = float(confidence)
                    results.append(enhanced_job)
                    debug_log(f"[{request_id}] - Enhanced job details:")
                    debug_log(
                        f"[{request_id}]   * Required skills: {len(enhanced_job.get('required_skills', []))}")
                    debug_log(
                        f"[{request_id}]   * Matching skills: {len(enhanced_job.get('matching_skills', []))}")
                    debug_log(
                        f"[{request_id}]   * Description length: {len(enhanced_job.get('description', ''))}")

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
                f"[{request_id}] ===== Completed Text-Based Recommendation Request =====\n")
            return jsonify(response)

        except Exception as e:
            debug_log(
                f"[{request_id}] ERROR in text-based recommendation: {str(e)}")
            debug_log(
                f"[{request_id}] ===== Failed Text-Based Recommendation Request =====\n")
            return jsonify({
                'status': 'error',
                'request_id': request_id,
                'message': f'Error processing request: {str(e)}'
            }), 500

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
