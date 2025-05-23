"""
Production server runner for the Job Recommendation API
This runs without watchdog reloading which prevents constant rebuilding of the graph
"""
from src.main import create_app

if __name__ == '__main__':
    print("Starting Job Recommendation API in production mode...")
    app = create_app()
    # Run without debug mode and reloader to prevent constant rebuilding of the graph
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
