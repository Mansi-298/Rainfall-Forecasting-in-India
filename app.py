# app.py - Fixed Flask Frontend for Rainfall Prediction

from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Check if model files exist
model_path = 'models/rainfall_model.pkl'
scaler_path = 'models/scaler.pkl'
features_path = 'models/feature_names.pkl'
encoder_path = 'models/label_encoder.pkl'

# Load the model and preprocessing components
try:
    print("Loading model and preprocessing components...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    # Load label encoder if it exists
    if os.path.exists(encoder_path):
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
    else:
        label_encoder = None
    
    print("Model and components loaded successfully")
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model_loaded = False

@app.route('/')
def home():
    if not model_loaded:
        return render_template('index.html', 
                            error="Model files not found. Please run rainfall_prediction.py first.")
    
    # Get the list of subdivisions if label encoder exists
    subdivisions = []
    if label_encoder is not None:
        subdivisions = label_encoder.classes_.tolist()
    
    return render_template('index.html', subdivisions=subdivisions)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return render_template('index.html', 
                            error="Model files not found. Please run rainfall_prediction.py first.")
    
    try:
        # Create a dictionary to store input features
        features = {}
        
        # Handle subdivision if it exists
        if 'SUBDIVISION' in request.form and label_encoder is not None:
            subdivision = request.form['SUBDIVISION']
            features['SUBDIVISION_CODE'] = label_encoder.transform([subdivision])[0]
        
        # Get numeric inputs
        for feature in feature_names:
            if feature in request.form:
                features[feature] = float(request.form[feature])
            elif feature != 'SUBDIVISION_CODE':  # Skip if not in form and not subdivision code
                features[feature] = 0
        
        # Create DataFrame with the input features
        input_df = pd.DataFrame([features])
        
        # Ensure all needed features are present
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match the training data
        input_df = input_df[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get subdivision name if applicable
        subdivision_name = None
        if 'SUBDIVISION' in request.form and label_encoder is not None:
            subdivision_name = request.form['SUBDIVISION']
        
        return render_template('index.html', 
                            prediction=f'Predicted Annual Rainfall: {prediction:.2f} mm',
                            input_data=request.form,
                            subdivisions=label_encoder.classes_.tolist() if label_encoder else None,
                            selected_subdivision=subdivision_name)
    
    except Exception as e:
        return render_template('index.html', 
                            error=f'Error making prediction: {str(e)}',
                            subdivisions=label_encoder.classes_.tolist() if label_encoder else None)

if __name__ == '__main__':
    app.run(debug=True)