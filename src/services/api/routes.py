import uuid
import json
import time
import os
import csv
from flask import jsonify, request

from src.config import DEFAULT_TOP_K
from .job_recommendation_service import JobRecommendationService
from src.utils.helpers import debug_log
from src.modeling.predict_dual import get_predictor

# Define project root for file paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_salary_data():
    """
    Load salary data with case-insensitive lookup support
    
    Returns:
        tuple: (salary_data dict, salary_data_lower dict for case-insensitive lookup)
    """
    salary_data = {}
    salary_data_lower = {}  # Case-insensitive lookup dictionary
    salary_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'salary.csv')
    
    try:
        with open(salary_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                job_title = row['Job Title'].strip('"')
                salary = row['Average Salary (USD)'].strip('"')
                salary_data[job_title] = salary
                # Create case-insensitive lookup
                salary_data_lower[job_title.lower()] = salary
    except Exception as e:
        debug_log(f"Error loading salary data: {str(e)}")
        
    return salary_data, salary_data_lower

def get_salary_for_job(job_title, salary_data, salary_data_lower):
    """
    Get salary for a job title with case-insensitive matching
    
    Args:
        job_title: Job title to look up
        salary_data: Dictionary of job titles to salaries
        salary_data_lower: Dictionary of lowercase job titles to salaries
        
    Returns:
        str: Salary for the job title or default message
    """
    # Try direct lookup first
    salary = salary_data.get(job_title, None)
    if not salary:
        # Try case-insensitive lookup
        salary = salary_data_lower.get(job_title.lower(), "Salary data not available")
    return salary

def register_routes(app, model_manager):
    """Register all API routes"""
    # Initialize the job recommendation service
    job_recommendation_service = JobRecommendationService()

    # Initialize the job recommendation model once at startup with timing measurement
    start_time = time.time()
    debug_log("Starting job recommendation model initialization...")
    success = job_recommendation_service.initialize_new_model()
    end_time = time.time()
    debug_log(
        f"Job recommendation model initialization {'completed successfully' if success else 'failed'} in {end_time - start_time:.2f} seconds")
    
    # Initialize the job prediction model (Dual Ensemble)
    debug_log("Starting job prediction model initialization...")
    predictor = get_predictor()
    debug_log("Job prediction model initialization completed")

    @app.route('/', methods=['GET'])
    def index():
        """Root endpoint with API information"""
        return jsonify({
            'status': 'success',
            'message': 'CareerConnect API is running',
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
                    'description': 'Get job recommendations from text description',
                    'example_payload': {
                        'text': 'I am a data scientist with expertise in Python, SQL, and machine learning...',
                        'top_n': 5
                    }
                },
                {
                    'path': '/api/recommend-GNN-Onet-from-cv',
                    'method': 'POST',
                    'description': 'Get job recommendations from CV (PDF)',
                    'notes': 'Accepts direct PDF file upload through multipart/form-data',
                    'example_payload': 'Form data with file field named "file" and optional "top_n" parameter'
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
                },
                {
                    'path': '/api/predict/text',
                    'method': 'POST',
                    'description': 'Predict job roles from the provided text',
                    'example_payload': {
                        'text': 'I am a software developer with expertise in Python, JavaScript, and React...',
                        'top_k': 3
                    }
                },
                {
                    'path': '/api/predict/technologies',
                    'method': 'POST',
                    'description': 'Predict job roles from a list of technologies',
                    'example_payload': {
                        'technologies': ['Python', 'JavaScript', 'React'],
                        'top_k': 3
                    }
                },
                {
                    'path': '/api/health',
                    'method': 'GET',
                    'description': 'Health check endpoint to verify if the API is running'
                }
            ]
        })

    @app.route('/api/recommend-GNN-Onet', methods=['POST'])
    def recommend_jobs_new():
        """Recommend jobs based on skills and technologies using the new model"""
        request_id = str(uuid.uuid4())[:8]

        # Parse request data
        if request.content_type and 'application/json' in request.content_type:
            data = request.get_json(force=True, silent=True)
        else:
            try:
                data = json.loads(request.data.decode('utf-8'))
            except:
                data = None

        # Call service method to get recommendations
        result = job_recommendation_service.recommend_jobs_from_skills(
            data, request_id)

        # Check if the result is a tuple (response, status_code)
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        else:
            return jsonify(result)

    @app.route('/api/analyze/onet', methods=['POST'])
    def recommend_from_text_new():
        """Recommend jobs based on text description using the new model"""
        request_id = str(uuid.uuid4())[:8]

        # Parse request data
        if request.content_type and 'application/json' in request.content_type:
            data = request.get_json(force=True, silent=True)
        else:
            try:
                data = json.loads(request.data.decode('utf-8'))
            except:
                data = None

        # Call service method to get recommendations
        result = job_recommendation_service.recommend_from_text(
            data, request_id)

        # Check if the result is a tuple (response, status_code)
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        else:
            return jsonify(result)

    @app.route('/api/analyze/onet/multipart/form-data', methods=['POST'])
    def recommend_from_cv_new():
        """Recommend jobs based on CV (PDF) using the new model"""
        request_id = str(uuid.uuid4())[:8]

        # Check if there's a file in the request
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file uploaded. Please upload a PDF file using multipart/form-data with a field named "file".'
            }), 400

        # Get the uploaded file
        pdf_file = request.files['file']

        # Check if the file has a name and is a PDF
        if pdf_file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400

        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({
                'status': 'error',
                'message': 'Uploaded file must be a PDF'
            }), 400

        # Get top_n parameter from form or query parameters
        top_n = DEFAULT_TOP_K
        if 'top_k' in request.form:
            top_k = int(request.form.get('top_k'))
        elif 'top_k' in request.args:
            top_k = int(request.args.get('top_k'))

        # Call service method to get recommendations
        result = job_recommendation_service.recommend_from_cv(
            pdf_file, top_k, request_id)

        # Check if the result is a tuple (response, status_code)
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        else:
            return jsonify(result)

    # Job prediction endpoints (migrated from FastAPI)
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint to verify if the API is running."""
        return jsonify({"status": "ok"})

    @app.route('/api/analyze/stackoverflow', methods=['POST'])
    def predict_from_text():
        """Predict job roles from the provided text.
        
        Returns the top predictions with salary information.
        """
        start_time = time.time()
        try:
            # Parse request data
            if request.content_type and 'application/json' in request.content_type:
                data = request.get_json(force=True, silent=True)
            else:
                try:
                    data = json.loads(request.data.decode('utf-8'))
                except:
                    return jsonify({
                        "status": "error",
                        "message": "Invalid JSON data"
                    }), 400
            
            if not data or 'text' not in data:
                return jsonify({
                    "status": "error",
                    "message": "Missing required field: text"
                }), 400
                
            top_k = data.get('top_k', 3)
            
            # Get predictor and make prediction
            predictor = get_predictor()
            result = predictor.predict_from_text(data['text'], top_k=top_k)
            
            # Load salary data
            salary_data, salary_data_lower = load_salary_data()
            
            # Format the response
            formatted_results = []
            for pred in result["ensemble_predictions"]:
                job_title = pred['role']
                # Find salary for this job title or use a default value
                salary = get_salary_for_job(job_title, salary_data, salary_data_lower)
                
                formatted_results.append({
                    "title": job_title,
                    "matchScore": round(pred['probability'] * 100, 2),  # Convert to percentage (0-100)
                    "keySkills": result["extracted_technologies"],
                    "salary": salary
                })
            
            processing_time = time.time() - start_time
            debug_log(f"Processed text in {processing_time:.2f} seconds")
            
            return jsonify(formatted_results)
            
        except Exception as e:
            debug_log(f"Error in predict_from_text: {str(e)}")
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500

    @app.route('/api/analyze/stackoverflow/multipart/form-data', methods=['POST'])
    def predict_from_cv():
        """Predict job roles from a CV (PDF file).
        
        Returns the top predictions with salary information.
        """
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        try:
            # Check if there's a file in the request
            if 'file' not in request.files:
                return jsonify({
                    'status': 'error',
                    'message': 'No file provided'
                }), 400
                
            # Get the uploaded file
            pdf_file = request.files['file']
            
            # Check if the file has a name and is a PDF
            if pdf_file.filename == '':
                return jsonify({
                    'status': 'error',
                    'message': 'No file selected'
                }), 400
                
            if not pdf_file.filename.lower().endswith('.pdf'):
                return jsonify({
                    'status': 'error',
                    'message': 'Only PDF files are supported'
                }), 400
                
            # Get the top_k parameter from form or query parameters
            top_k = 3  # Default value
            if 'top_k' in request.form:
                top_k = int(request.form.get('top_k'))
            elif 'top_k' in request.args:
                top_k = int(request.args.get('top_k'))
            
            debug_log(f"[{request_id}] Processing uploaded CV: {pdf_file.filename}")
            
            # Make prediction using the PDF file
            predictor = get_predictor()
            result = predictor.predict_from_pdf(pdf_file, top_k=top_k)
            
            # Check if there was an error
            if 'status' in result and result['status'] == 'error':
                return jsonify(result), 400
                
            # Load salary data
            salary_data, salary_data_lower = load_salary_data()
            
            # Format the response
            formatted_results = []
            for pred in result["ensemble_predictions"]:
                job_title = pred['role']
                # Find salary for this job title or use a default value
                salary = get_salary_for_job(job_title, salary_data, salary_data_lower)
                
                formatted_results.append({
                    "title": job_title,
                    "matchScore": round(pred['probability'] * 100, 2),  # Convert to percentage (0-100)
                    "keySkills": result["extracted_technologies"],
                    "salary": salary
                })
            
            processing_time = time.time() - start_time
            debug_log(f"[{request_id}] Processed CV in {processing_time:.2f} seconds")
            
            return jsonify(formatted_results)
            
        except Exception as e:
            debug_log(f"Error in predict_from_cv: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'An error occurred: {str(e)}'
            }), 500

    @app.route('/api/predict/technologies', methods=['POST'])
    def predict_from_technologies():
        """Predict job roles from a list of technologies.
        
        Returns the top predictions with salary information.
        """
        start_time = time.time()
        try:
            # Parse request data
            if request.content_type and 'application/json' in request.content_type:
                data = request.get_json(force=True, silent=True)
            else:
                try:
                    data = json.loads(request.data.decode('utf-8'))
                except:
                    return jsonify({
                        "status": "error",
                        "message": "Invalid JSON data"
                    }), 400
            
            if not data or 'technologies' not in data:
                return jsonify({
                    "status": "error",
                    "message": "Missing required field: technologies"
                }), 400
                
            top_k = data.get('top_k', 3)
            
            # Get predictor and make prediction
            predictor = get_predictor()
            result = predictor.predict_from_technologies(data['technologies'], top_k=top_k)
            
            # Load salary data
            salary_data, salary_data_lower = load_salary_data()
            
            # Format the response
            formatted_results = []
            for pred in result["ensemble_predictions"]:
                job_title = pred['role']
                # Find salary for this job title or use a default value
                salary = get_salary_for_job(job_title, salary_data, salary_data_lower)
                
                formatted_results.append({
                    "title": job_title,
                    "matchScore": round(pred['probability'] * 100, 2),  # Convert to percentage (0-100)
                    "keySkills": result["extracted_technologies"] if "extracted_technologies" in result else data['technologies'],
                    "salary": salary
                })
            
            processing_time = time.time() - start_time
            debug_log(f"Processed technologies in {processing_time:.2f} seconds")
            
            return jsonify(formatted_results)
            
        except Exception as e:
            debug_log(f"Error in predict_from_technologies: {str(e)}")
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500
    
    # Return the app with registered routes
    return app
