from flask import Flask, render_template, jsonify, request, redirect, url_for, session
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import json
import logging
import sys
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import re
from sklearn.linear_model import LinearRegression, ElasticNet, RidgeCV, LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key in production
app.permanent_session_lifetime = timedelta(days=365*100)  # Set session lifetime to 100 years (effectively permanent)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///steel_plant.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    last_login = db.Column(db.DateTime(timezone=True))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Create database tables
with app.app_context():
    db.create_all()
    logger.info("Database tables created successfully")

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        if 'user_id' in session:
            return redirect(url_for('home'))
        return render_template('register.html')
    
    try:
        if not request.is_json:
            logger.error("Registration request did not contain JSON data")
            return jsonify({
                'success': False,
                'message': 'Invalid request format. Please send JSON data.'
            }), 400

        data = request.get_json()
        
        # Validate required fields
        required_fields = ['email', 'password', 'fullName']
        if not all(field in data for field in required_fields):
            missing_fields = [field for field in required_fields if field not in data]
            logger.error(f"Missing required fields in registration: {missing_fields}")
            return jsonify({
                'success': False,
                'message': 'Missing required fields: ' + ', '.join(missing_fields)
            }), 400

        email = data['email'].lower().strip()
        password = data['password']
        full_name = data['fullName'].strip()

        # Validate email format
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            logger.error(f"Invalid email format: {email}")
            return jsonify({
                'success': False,
                'message': 'Invalid email format'
            }), 400

        # Check if email already exists
        if User.query.filter_by(email=email).first():
            logger.warning(f"Attempted registration with existing email: {email}")
            return jsonify({
                'success': False,
                'message': 'Email already registered'
            }), 400

        # Validate password length
        if len(password) < 6:
            logger.error("Password too short during registration")
            return jsonify({
                'success': False,
                'message': 'Password must be at least 6 characters long'
            }), 400

        # Create new user
        new_user = User(email=email, full_name=full_name)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()

        logger.info(f"Successfully registered new user: {email}")
        return jsonify({
            'success': True,
            'message': 'Registration successful'
        })

    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': 'An error occurred during registration'
        }), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received in login request")
            return jsonify({'success': False, 'message': 'Invalid request data'}), 400

        email = data.get('email')
        password = data.get('password')

        # Validate required fields
        if not all([email, password]):
            logger.error("Missing required fields in login request")
            return jsonify({'success': False, 'message': 'Email and password are required'}), 400

        user = User.query.filter_by(email=email.lower().strip()).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session.permanent = True  # Always set session as permanent
            
            # Update last login timestamp
            user.last_login = func.now()
            db.session.commit()
            
            logger.info(f"User logged in successfully: {email}")
            return jsonify({'success': True, 'message': 'Login successful'})

        logger.warning(f"Failed login attempt for email: {email}")
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        return jsonify({'success': False, 'message': 'Login failed'}), 500

@app.route('/logout')
def logout():
    user_id = session.pop('user_id', None)
    if user_id:
        logger.info(f"User logged out: {user_id}")
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/models')
@login_required
def models():
    return render_template('models.html')

# Load and preprocess data
def load_data():
    logger.info("Loading data files...")
    try:
        blast_data = pd.read_csv('blast_data.csv')
        sinter_data = pd.read_csv('blast_data1.csv')
        knowledge_base = pd.read_csv('blast_data2.csv', 
                                   sep=',',
                                   quoting=1,
                                   doublequote=True,
                                   engine='python')
        
        # Ensure numeric columns in sinter_data are properly converted
        numeric_columns = ['basicity', 'bed_height', 'coke_rate', 'return_fines_ratio',
                         'sio2_content', 'al2o3_content', 'mgo_content',
                         'ignition_temperature', 'burn_through_temperature',
                         'moisture_content', 'tumbler_index', 'reducibility_index',
                         'productivity']
        
        for col in numeric_columns:
            if col in sinter_data.columns:
                sinter_data[col] = pd.to_numeric(sinter_data[col], errors='coerce')
        
        # Remove any rows with NaN values after conversion
        sinter_data = sinter_data.dropna()
        
        logger.info(f"Sinter data shape: {sinter_data.shape}")
        logger.info(f"Sinter data columns: {sinter_data.columns.tolist()}")
        logger.info(f"Sinter data summary:\n{sinter_data.describe()}")
        
        return blast_data, sinter_data, knowledge_base
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

# Initialize models with performance tracking
def create_models(blast_data, sinter_data):
    logger.info("Creating models...")
    
    # Blast Furnace Model
    X_blast = blast_data[['blast_temperature', 'oxygen_enrichment', 'coal_injection_rate', 
                         'top_pressure', 'moisture', 'ore_fe_content']]
    y_blast = blast_data['hot_metal_temperature']
    
    # Split data for blast furnace
    X_blast_train, X_blast_test, y_blast_train, y_blast_test = train_test_split(
        X_blast, y_blast, test_size=0.2, random_state=42
    )
    
    # Create and train final blast furnace model with optimized parameters
    blast_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_split=4,
        min_samples_leaf=3,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
    
    logger.info("Training Blast Furnace Model...")
    blast_model.fit(X_blast_train, y_blast_train)
    
    # Calculate blast furnace model performance
    blast_train_pred = blast_model.predict(X_blast_train)
    blast_test_pred = blast_model.predict(X_blast_test)
    
    blast_performance = {
        'train_r2': r2_score(y_blast_train, blast_train_pred),
        'test_r2': r2_score(y_blast_test, blast_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_blast_train, blast_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_blast_test, blast_test_pred)),
        'train_mae': mean_absolute_error(y_blast_train, blast_train_pred),
        'test_mae': mean_absolute_error(y_blast_test, blast_test_pred),
        'feature_importance': dict(zip(X_blast.columns, blast_model.feature_importances_)),
        'actual_vs_predicted': {
            'train': {'actual': y_blast_train.tolist(), 'predicted': blast_train_pred.tolist()},
            'test': {'actual': y_blast_test.tolist(), 'predicted': blast_test_pred.tolist()}
        }
    }

    logger.info(f"Blast Furnace Model Performance - Train R²: {blast_performance['train_r2']:.4f}, Test R²: {blast_performance['test_r2']:.4f}")
    
    # Enhanced Sinter Plant Model
    logger.info("\nInitializing Sinter Plant Model...")
    
    # Use domain knowledge for feature selection and constraints
    primary_features = ['basicity', 'bed_height', 'coke_rate', 'return_fines_ratio']
    secondary_features = ['sio2_content', 'al2o3_content', 'mgo_content', 
                         'ignition_temperature', 'burn_through_temperature', 
                         'moisture_content']
    quality_indicators = ['tumbler_index', 'reducibility_index']
    
    # Ensure all required columns exist
    missing_columns = [col for col in primary_features + secondary_features + quality_indicators 
                      if col not in sinter_data.columns]
    if missing_columns:
        logger.warning(f"Missing columns in sinter data: {missing_columns}")
        # Remove missing columns from feature lists
        primary_features = [col for col in primary_features if col in sinter_data.columns]
        secondary_features = [col for col in secondary_features if col in sinter_data.columns]
        quality_indicators = [col for col in quality_indicators if col in sinter_data.columns]

    X_sinter = sinter_data[primary_features + secondary_features + quality_indicators]
    y_sinter = sinter_data['productivity']
    
    # Log dataset characteristics
    logger.info("\nSinter Plant Data Characteristics:")
    logger.info(f"Number of samples: {len(X_sinter)}")
    logger.info(f"Number of features: {X_sinter.shape[1]}")
    logger.info(f"Features used: {X_sinter.columns.tolist()}")
    
    # Validate against domain knowledge
    logger.info("\nValidating feature ranges based on domain knowledge...")
    
    # Check basicity range (1.9-2.2)
    basicity_mask = (X_sinter['basicity'] >= 1.9) & (X_sinter['basicity'] <= 2.2)
    logger.info(f"Samples within optimal basicity range: {basicity_mask.sum()}/{len(X_sinter)}")
    
    # Check bed height range (520-580 mm)
    height_mask = (X_sinter['bed_height'] >= 520) & (X_sinter['bed_height'] <= 580)
    logger.info(f"Samples within optimal bed height range: {height_mask.sum()}/{len(X_sinter)}")
    
    # Check coke rate range (7.0-8.0%)
    coke_mask = (X_sinter['coke_rate'] >= 7.0) & (X_sinter['coke_rate'] <= 8.0)
    logger.info(f"Samples within optimal coke rate range: {coke_mask.sum()}/{len(X_sinter)}")
    
    # Check return fines ratio (< 30%)
    fines_mask = X_sinter['return_fines_ratio'] < 30
    logger.info(f"Samples within optimal return fines range: {fines_mask.sum()}/{len(X_sinter)}")
    
    # Create domain knowledge based features
    X_sinter_extended = X_sinter.copy()
    
    # Thermal process interactions
    X_sinter_extended['thermal_ratio'] = X_sinter['burn_through_temperature'] / X_sinter['ignition_temperature']
    
    # Material quality interactions
    X_sinter_extended['strength_index'] = X_sinter['tumbler_index'] * (1 - X_sinter['return_fines_ratio']/100)
    
    # Chemical composition effect
    X_sinter_extended['gangue_ratio'] = (X_sinter['sio2_content'] + X_sinter['al2o3_content']) / X_sinter['basicity']
    
    # Standardize features
    scaler = StandardScaler()
    X_sinter_scaled = scaler.fit_transform(X_sinter_extended)
    
    # For very small datasets, use Leave-One-Out Cross-Validation
    cv = LeaveOneOut()
    logger.info("\nUsing Leave-One-Out Cross-Validation due to small dataset size")
    
    # Initialize models with strong regularization
    models = {
        'ridge': RidgeCV(
            alphas=[0.1, 1.0, 10.0, 100.0, 1000.0],
            scoring='neg_mean_squared_error'
        ),
        'elastic_net': ElasticNet(
            alpha=1.0,
            l1_ratio=0.5,
            random_state=42,
            max_iter=2000
        )
    }
    
    # Evaluate models
    logger.info("\nCross-validation Results:")
    best_model = None
    best_rmse = float('inf')
    cv_results = {}
    model_names = {
        'RidgeCV': 'ridge',
        'ElasticNet': 'elastic_net'
    }
    
    for name, model in models.items():
        try:
            # Perform Leave-One-Out cross-validation
            predictions = []
            actuals = []
            
            for train_idx, test_idx in cv.split(X_sinter_scaled):
                X_train, X_test = X_sinter_scaled[train_idx], X_sinter_scaled[test_idx]
                y_train, y_test = y_sinter.iloc[train_idx], y_sinter.iloc[test_idx]
                
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                predictions.append(pred[0])
                actuals.append(y_test.iloc[0])
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            r2 = r2_score(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            
            cv_results[name] = {
                'mean_r2': r2,
                'mean_rmse': rmse,
                'mean_mae': mae,
                'predictions': predictions,
                'actuals': actuals
            }
            
            logger.info(f"\n{name.upper()}:")
            logger.info(f"LOOCV R²: {r2:.4f}")
            logger.info(f"LOOCV RMSE: {rmse:.4f}")
            logger.info(f"LOOCV MAE: {mae:.4f}")
            
            if rmse < best_rmse and rmse > 1e-6:
                best_rmse = rmse
                best_model = model
                
        except Exception as e:
            logger.warning(f"Error with {name}: {str(e)}")
    
    # Train final model on full dataset
    if best_model is None:
        logger.warning("No model performed well, using Ridge regression with high regularization")
        best_model = RidgeCV(alphas=[1000.0])
        model_name = 'ridge'
    else:
        model_name = model_names.get(type(best_model).__name__, 'ridge')
    
    sinter_model = best_model
    sinter_model.fit(X_sinter_scaled, y_sinter)
    
    # Get predictions for all samples
    y_pred = sinter_model.predict(X_sinter_scaled)
    
    # Calculate final metrics
    final_rmse = np.sqrt(mean_squared_error(y_sinter, y_pred))
    final_r2 = r2_score(y_sinter, y_pred)
    final_mae = mean_absolute_error(y_sinter, y_pred)
    
    logger.info("\nFinal Model Performance:")
    logger.info(f"R² Score: {final_r2:.4f}")
    logger.info(f"RMSE: {final_rmse:.4f}")
    logger.info(f"MAE: {final_mae:.4f}")
    
    # Get feature importance
    if hasattr(best_model, 'coef_'):
        importance = np.abs(best_model.coef_)
    else:
        importance = np.ones(len(X_sinter.columns)) / len(X_sinter.columns)
    
    feature_importance = dict(zip(X_sinter.columns, importance))
    logger.info("\nFeature Importance:")
    for feature, imp in sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True):
        logger.info(f"{feature}: {imp:.4f}")
    
    # Store performance metrics using LOOCV results for test metrics
    best_cv_results = cv_results[model_name]
    sinter_performance = {
        'train_r2': final_r2,
        'test_r2': best_cv_results['mean_r2'],
        'train_rmse': final_rmse,
        'test_rmse': best_cv_results['mean_rmse'],
        'train_mae': final_mae,
        'test_mae': best_cv_results['mean_mae'],
        'feature_importance': feature_importance,
        'actual_vs_predicted': {
            'train': {'actual': y_sinter.tolist(), 'predicted': y_pred.tolist()},
            'test': {'actual': best_cv_results['actuals'], 'predicted': best_cv_results['predictions']}
        },
        'cv_results': cv_results
    }
    
    return blast_model, sinter_model, X_blast.columns, X_sinter.columns, blast_performance, sinter_performance

# Load data and create models - Move this outside of if __name__ == '__main__':
logger.info("Loading data and initializing models...")
blast_data, sinter_data, knowledge_base = load_data()
blast_model, sinter_model, blast_features, sinter_features, blast_performance, sinter_performance = create_models(blast_data, sinter_data)
logger.info("Models initialized successfully")

@app.route('/api/blast-furnace/predict', methods=['POST'])
def predict_blast_furnace():
    try:
        data = request.json
        input_data = np.array([[
            float(data['blast_temperature']),
            float(data['oxygen_enrichment']),
            float(data['coal_injection_rate']),
            float(data['top_pressure']),
            float(data['moisture']),
            float(data['ore_fe_content'])
        ]])
        
        logger.info(f"Received prediction request with input: {input_data}")
        
        # Validate input ranges
        input_ranges = {
            'blast_temperature': (1000, 1300),
            'oxygen_enrichment': (0, 10),
            'coal_injection_rate': (100, 250),
            'top_pressure': (1, 3),
            'moisture': (15, 30),
            'ore_fe_content': (55, 65)
        }
        
        for i, (feature, value) in enumerate(zip(input_ranges.keys(), input_data[0])):
            min_val, max_val = input_ranges[feature]
            if value < min_val or value > max_val:
                logger.warning(f"Input value for {feature} ({value}) is outside expected range [{min_val}, {max_val}]")
                return jsonify({
                    'error': f"Input value for {feature} ({value}) is outside expected range [{min_val}, {max_val}]"
                }), 400
        
        prediction = blast_model.predict(input_data)[0]
        logger.info(f"Prediction result: {prediction:.2f}°C")
        
        return jsonify({'predicted_temperature': float(prediction)})
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sinter-plant/predict', methods=['POST'])
def predict_sinter_plant():
    try:
        data = request.json
        logger.info(f"Received prediction request with data: {data}")
        
        input_data = np.array([[
            float(data['basicity']),
            float(data['sio2_content']),
            float(data['al2o3_content']),
            float(data['mgo_content']),
            float(data['bed_height']),
            float(data['ignition_temperature'])
        ]])
        
        logger.info(f"Processed input data: {input_data}")
        
        prediction = sinter_model.predict(input_data)[0]
        logger.info(f"Model prediction: {prediction:.4f}")
        
        return jsonify({'predicted_productivity': float(prediction)})
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/blast-furnace/data')
def get_blast_furnace_data():
    return jsonify(blast_data.to_dict(orient='records'))

@app.route('/api/sinter-plant/data')
def get_sinter_plant_data():
    return jsonify(sinter_data.to_dict(orient='records'))

@app.route('/api/knowledge/search', methods=['POST'])
def search_knowledge():
    query = request.json.get('query', '').lower()
    if not query:
        return jsonify({'results': []})
    
    # Search in knowledge base
    results = []
    for _, row in knowledge_base.iterrows():
        if query in row['content'].lower():
            results.append({
                'title': row['file_name'],
                'category': row['category'],
                'content': row['content']
            })
    
    return jsonify({'results': results})

@app.route('/api/charts/moisture-data')
def get_moisture_data():
    # Generate sample moisture data
    times = pd.date_range(start='2024-01-01 06:00:00', periods=7, freq='H')
    moisture_data = {
        'times': [t.strftime('%H:%M') for t in times],
        'moisture': [10.2, 10.0, 9.9, 9.7, 9.8, 9.8, 9.7],
        'target': [9.5] * 7
    }
    return jsonify(moisture_data)

@app.route('/api/charts/filtration-data')
def get_filtration_data():
    # Generate sample filtration data
    times = pd.date_range(start='2024-01-01 06:00:00', periods=7, freq='H')
    filtration_data = {
        'times': [t.strftime('%H:%M') for t in times],
        'filtration_rate': [81.0, 81.8, 82.5, 83.5, 84.0, 82.8, 82.0],
        'solids_recovery': [98.5, 98.65, 98.8, 99.0, 99.1, 99.0, 98.9]
    }
    return jsonify(filtration_data)

@app.route('/api/charts/feature-importance')
def get_feature_importance():
    # Feature importance data for pellet production
    feature_data = {
        'features': ['Induration Temp', 'Induration Time', 'Bentonite Addition', 
                    'Moisture Content', 'Concentrate Fe', 'Disc Speed'],
        'importance': [0.32, 0.22, 0.18, 0.12, 0.09, 0.07]
    }
    return jsonify(feature_data)

@app.route('/api/models/performance')
def get_models_performance():
    return jsonify({
        'blast_furnace': blast_performance,
        'sinter_plant': sinter_performance
    })

@app.route('/api/models/residuals')
def get_residuals():
    # Calculate residuals for both models
    blast_residuals = {
        'train': np.subtract(
            blast_performance['actual_vs_predicted']['train']['actual'],
            blast_performance['actual_vs_predicted']['train']['predicted']
        ).tolist(),
        'test': np.subtract(
            blast_performance['actual_vs_predicted']['test']['actual'],
            blast_performance['actual_vs_predicted']['test']['predicted']
        ).tolist()
    }
    
    sinter_residuals = {
        'train': np.subtract(
            sinter_performance['actual_vs_predicted']['train']['actual'],
            sinter_performance['actual_vs_predicted']['train']['predicted']
        ).tolist(),
        'test': np.subtract(
            sinter_performance['actual_vs_predicted']['test']['actual'],
            sinter_performance['actual_vs_predicted']['test']['predicted']
        ).tolist()
    }
    
    return jsonify({
        'blast_furnace': blast_residuals,
        'sinter_plant': sinter_residuals
    })

if __name__ == '__main__':
    app.run(debug=True) 