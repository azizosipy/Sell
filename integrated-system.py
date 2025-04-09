"""
JSW Toranagallu Steel Plant - Comprehensive Integrated Model

This is the main integration file that brings together all components of the
JSW Steel Plant Digital Twin system:
- Configuration
- Data Generation
- ML/DL Models
- Dashboard
- API Services

Running this file will initialize the complete system.

Author: Claude AI
Date: April 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import logging
import time
import matplotlib.pyplot as plt
import datetime
import threading
import webbrowser
import base64
import argparse
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.serving import run_simple
from werkzeug.middleware.dispatcher import DispatcherMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("JSW_Integrated_System")

def setup_environment():
    """Ensure all required dependencies are available"""
    try:
        import tensorflow as tf
        import torch
        import dash
        import dash_bootstrap_components as dbc
        import plotly.graph_objects as go
        import cv2
        import sklearn
        import joblib
        import transformers
        import faiss
        
        logger.info("All dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install all required packages before running the system")
        return False

class IntegratedSystem:
    """Main class that integrates all components of the system"""
    
    def __init__(self, args):
        """Initialize the integrated system"""
        self.args = args
        
        # Initialize system components
        from config import Config
        self.config = Config()
        
        # Create directories
        self._create_directories()
        
        # Initialize components
        self._init_components()
    
    def _create_directories(self):
        """Ensure all required directories exist"""
        # Make sure base directory exists
        os.makedirs(self.config.base_dir, exist_ok=True)
        
        # Create web directory and add necessary files
        web_dir = os.path.join(self.config.web_dir, 'templates')
        os.makedirs(web_dir, exist_ok=True)
        
        # Create a simple index.html file if it doesn't exist
        index_path = os.path.join(web_dir, 'index.html')
        if not os.path.exists(index_path):
            with open(index_path, 'w') as f:
                f.write('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>JSW Steel Plant Comprehensive System</title>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                    <style>
                        body { padding-top: 20px; }
                        .card { margin-bottom: 20px; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <header class="mb-4">
                            <h1 class="text-primary">JSW Steel Plant</h1>
                            <h4 class="text-muted">Comprehensive Digital Twin System</h4>
                        </header>
                        
                        <div class="row">
                            <div class="col-md-12">
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">System Dashboard</h5>
                                        <p class="card-text">Access the interactive dashboard for monitoring and prediction.</p>
                                        <a href="/dashboard/" class="btn btn-primary">Open Dashboard</a>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">API Documentation</h5>
                                        <p class="card-text">View API documentation and endpoints.</p>
                                        <a href="/api/docs" class="btn btn-secondary">View API Docs</a>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">System Status</h5>
                                        <p class="card-text">View system status and component health.</p>
                                        <a href="/status" class="btn btn-info">Check Status</a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
                </body>
                </html>
                ''')
    
    def _init_components(self):
        """Initialize all system components"""
        logger.info("Initializing all system components...")
        
        # Initialize data generator
        from comprehensive_synthetic_data_generator import ComprehensiveSyntheticDataGenerator
        self.data_generator = ComprehensiveSyntheticDataGenerator(self.config)
        
        # Initialize model builder
        from ml_model_builder import MLModelBuilder
        self.model_builder = MLModelBuilder(self.config)
        
        # Initialize dashboard integration
        from dashboard_integration import DashboardIntegration
        self.dashboard = DashboardIntegration(self.config, self.model_builder)
        
        logger.info("All components initialized successfully")
    
    def generate_data(self):
        """Generate synthetic data for all processes"""
        logger.info("Generating synthetic data...")
        self.data_generator.generate_all_data()
        logger.info("Data generation complete")
    
    def build_models(self):
        """Build and train all ML/DL models"""
        logger.info("Building machine learning models...")
        self.model_builder.build_all_models()
        logger.info("Model building complete")
    
    def run_dashboard(self):
        """Run the dashboard application"""
        host = self.args.host if hasattr(self.args, 'host') else '0.0.0.0'
        port = self.args.port if hasattr(self.args, 'port') else 8050
        debug = self.args.debug if hasattr(self.args, 'debug') else False
        
        logger.info(f"Starting dashboard on http://{host}:{port}/")
        
        # Open browser after a delay
        if not debug and host in ['0.0.0.0', '127.0.0.1', 'localhost']:
            threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}/")).start()
        
        # Run the server
        self.dashboard.flask_app.run(host=host, port=port, debug=debug)
    
    def run_system(self):
        """Run the complete integrated system"""
        # Check if data exists
        if self.args.generate_data:
            self.generate_data()
        
        # Check if models exist
        if self.args.build_models:
            self.build_models()
        
        # Run dashboard
        self.run_dashboard()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='JSW Steel Plant Comprehensive System')
    parser.add_argument('--generate-data', action='store_true', help='Generate synthetic data')
    parser.add_argument('--build-models', action='store_true', help='Build and train ML/DL models')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    return parser.parse_args()

def main():
    """Main entry point of the application"""
    # Check if environment is set up correctly
    if not setup_environment():
        logger.error("Environment setup failed. Exiting...")
        sys.exit(1)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create and run the integrated system
    system = IntegratedSystem(args)
    
    try:
        system.run_system()
    except KeyboardInterrupt:
        logger.info("System shutdown requested. Exiting...")
    except Exception as e:
        logger.error(f"Error running integrated system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()