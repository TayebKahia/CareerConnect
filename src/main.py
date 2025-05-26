import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from flask import Flask
import torch
from flask_cors import CORS

from models.mb.classes.model_loader import ModelManager
from src.services.api.routes import register_routes
from src.utils.helpers import debug_log


def create_app():
    """Create and configure the Flask application"""
    # Print device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize Flask app
    app = Flask(__name__)
    CORS(app)
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

    # Run with reduced reloader and no static file changes detection
    # Disable watching for file changes
    app.run(debug=False, host='0.0.0.0', port=8000, use_reloader=False)


if __name__ == '__main__':
    main()
