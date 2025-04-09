# Steel Plant Analytics Dashboard

This web application provides predictive analytics for blast furnace and sinter plant operations in a steel manufacturing facility. It uses machine learning models to predict hot metal temperature and sinter plant productivity based on various operational parameters.

## Features

- Blast Furnace Predictor: Predicts hot metal temperature based on operational parameters
- Sinter Plant Predictor: Predicts productivity based on sinter plant parameters
- Modern, responsive web interface
- Real-time predictions using machine learning models

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Use the forms to input parameters and get predictions:
   - Blast Furnace: Input operational parameters to predict hot metal temperature
   - Sinter Plant: Input sinter plant parameters to predict productivity

## Data Sources

The application uses the following data files:
- `blast_data.csv`: Blast furnace operational data
- `blast_data1.csv`: Sinter plant operational data
- `blast_data2.csv`: Additional operational parameters and documentation

## Model Details

- Blast Furnace Model: Random Forest Regressor trained on historical blast furnace data
- Sinter Plant Model: Random Forest Regressor trained on historical sinter plant data

## Requirements

- Python 3.7+
- Flask
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Dash
- Dash Bootstrap Components 