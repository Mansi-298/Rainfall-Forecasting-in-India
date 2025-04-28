# Fixed Rainfall Prediction in India
# Complete ML Pipeline with Decision Tree Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Create necessary directories
os.makedirs('models', exist_ok=True)

# Step 1: Data Loading
print("Step 1: Loading the rainfall dataset from Kaggle")
# For this example, we'll assume the dataset has been downloaded from Kaggle
# and is available in the working directory
df = pd.read_csv('data/rainfall in india 1901-2015.csv')

# Step 2: Data Exploration
print("\nStep 2: Exploring the dataset")
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows of the dataset:")
print(df.head())

print("\nData information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

print("\nChecking for missing values:")
print(df.isnull().sum())

# Step 3: Data Preprocessing
print("\nStep 3: Data preprocessing")

# Handle the SUBDIVISION column separately
# Convert SUBDIVISION to numeric using Label Encoding
if 'SUBDIVISION' in df.columns:
    print("Encoding the SUBDIVISION column...")
    label_encoder = LabelEncoder()
    df['SUBDIVISION_CODE'] = label_encoder.fit_transform(df['SUBDIVISION'])
    
    # Save the encoder for future use
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Remove the original text column
    df = df.drop('SUBDIVISION', axis=1)

# Fill missing values for numeric columns only
print("Handling missing values in numeric columns...")
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].mean())

# Step 4: Prepare Features and Target
print("\nStep 4: Preparing features and target")
# Features will be monthly rainfall, year, and subdivision code
# Target will be ANNUAL rainfall
X = df.drop(['ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec'], axis=1, errors='ignore')
y = df['ANNUAL']

print(f"Feature columns: {X.columns.tolist()}")
print(f"Target column: ANNUAL")

# Feature scaling
print("Applying feature scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train-Test Split
print("\nStep 5: Splitting data into training and testing sets")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Step 6: Model Building - Decision Tree Regressor
print("\nStep 6: Building Decision Tree Regression model")
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Step 7: Model Evaluation
print("\nStep 7: Evaluating the model")
y_pred = dt_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 8: Save the model and preprocessing components
print("\nStep 8: Saving the model for frontend application")
# Save the model
with open('models/rainfall_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

# Save the scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("Files saved to models/ directory:")
print("- models/rainfall_model.pkl")
print("- models/scaler.pkl")
print("- models/feature_names.pkl")
print("- models/label_encoder.pkl")

print("\nML pipeline completed successfully. The model is now ready to be used in the frontend application.")