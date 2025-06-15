import uuid
import json
from flask import jsonify, request

from src.config import DEFAULT_TOP_K
from .job_recommendation_service import JobRecommendationService
from src.utils.helpers import debug_log

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
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def register_routes(app, model_manager):
    """Register all API routes"""
    # Initialize the job recommendation service
    job_recommendation_service = JobRecommendationService()

    # Initialize the model once at startup with timing measurement
    import time
    start_time = time.time()
    debug_log("Starting model initialization...")
    success = job_recommendation_service.initialize_new_model()
    end_time = time.time()
    debug_log(
        f"Model initialization {'completed successfully' if success else 'failed'} in {end_time - start_time:.2f} seconds")

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
                    'path': '/api/predict-job',
                    'method': 'POST',
                    'description': 'Predict job based on text description using ESCO model',
                    'example_payload': {
                        'text': 'I am a software engineer with experience in Python and web development...',
                        'threshold': 0.5,
                        'similarity_threshold': 0.5,
                        'gcn_weight': 0.3
                    }
                },
                {
                    'path': '/api/debug-skills',
                    'method': 'POST',
                    'description': 'Debug skill extraction from text',
                    'example_payload': {
                        'text': 'I am proficient in Python and JavaScript...',
                        'similarity_threshold': 0.5
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

    @app.route('/api/recommend-GNN-Onet-from-text', methods=['POST'])
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

    @app.route('/api/recommend-GNN-Onet-from-cv', methods=['POST'])
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
        if 'top_n' in request.form:
            top_n = int(request.form.get('top_n'))
        elif 'top_n' in request.args:
            top_n = int(request.args.get('top_n'))

        # Call service method to get recommendations
        result = job_recommendation_service.recommend_from_cv(
            pdf_file, top_n, request_id)

        # Check if the result is a tuple (response, status_code)
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        else:
            return jsonify(result)

    # Add any additional endpoints below

    # Return the app with registered routes
    return app
