from flask import Flask
from pathlib import Path
from src.web.routes import register_routes

def create_app():
    # Set template folder path to the structured templates folder
    template_dir = Path(__file__).resolve().parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    
    # Register routes
    register_routes(app)
    
    return app
