from flask import Flask
import torch

from src.modeling.models.model_loader import ModelManager
from src.services.api.routes import register_routes
from src.utils.helpers import debug_log

def create_app():
    """Create and configure the Flask application"""
    # Print device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize Flask app
    app = Flask(__name__)
    
    # Initialize model manager
    model_manager = ModelManager()
    success = model_manager.initialize_model()
    
    if not success:
        debug_log("Failed to initialize model manager")
        raise RuntimeError("Failed to initialize model manager")
    
    # Register routes
    register_routes(app, model_manager)
    
    return app

def main():
    """Main entry point for the application"""
    print("Starting Job Recommendation API...")
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main() 