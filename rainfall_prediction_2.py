# Fixed Rainfall Prediction in India

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


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)  

# Step 1: Data Loading
print("Step 1: Loading the rainfall dataset from Kaggle")
df = pd.read_csv('data/rainfall in india 1901-2015.csv')

# Step 2: Data Exploration with Visualizations
print("\nStep 2: Exploring the dataset with visualizations")
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows of the dataset:")
print(df.head())

print("\nData information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

print("\nChecking for missing values:")
missing_values = df.isnull().sum()
print(missing_values)

# Visualization 1: Annual Rainfall Distribution
print("\nCreating visualization 1: Annual Rainfall Distribution")
plt.figure(figsize=(12, 6))
sns.histplot(df['ANNUAL'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Annual Rainfall across India (1901-2015)', fontsize=16)
plt.xlabel('Annual Rainfall (mm)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.savefig('plots/annual_rainfall_distribution.png')
plt.close()

# Visualization 2: Average Annual Rainfall by Subdivision
print("Creating visualization 2: Average Annual Rainfall by Subdivision")
plt.figure(figsize=(14, 10))
subdivision_rainfall = df.groupby('SUBDIVISION')['ANNUAL'].mean().sort_values(ascending=False)
sns.barplot(x=subdivision_rainfall.values, y=subdivision_rainfall.index, palette='viridis')
plt.title('Average Annual Rainfall by Subdivision', fontsize=16)
plt.xlabel('Average Annual Rainfall (mm)', fontsize=14)
plt.ylabel('Subdivision', fontsize=14)
plt.tight_layout()
plt.savefig('plots/avg_rainfall_by_subdivision.png')
plt.close()

# Visualization 3: Rainfall Trends Over Years
print("Creating visualization 3: Rainfall Trends Over Years")
yearly_avg = df.groupby('YEAR')['ANNUAL'].mean()
plt.figure(figsize=(14, 6))
sns.lineplot(x=yearly_avg.index, y=yearly_avg.values, marker='o', color='darkblue')
plt.title('Average Annual Rainfall Trend Over Years (1901-2015)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Rainfall (mm)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/rainfall_trend_over_years.png')
plt.close()

# Visualization 4: Seasonal Rainfall Distribution
print("Creating visualization 4: Seasonal Rainfall Distribution")
if all(col in df.columns for col in ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']):
    seasonal_data = pd.DataFrame({
        'Winter (Jan-Feb)': df['Jan-Feb'].mean(),
        'Summer (Mar-May)': df['Mar-May'].mean(),
        'Monsoon (Jun-Sep)': df['Jun-Sep'].mean(),
        'Post-Monsoon (Oct-Dec)': df['Oct-Dec'].mean()
    }, index=[0]).T.reset_index()
    seasonal_data.columns = ['Season', 'Average Rainfall']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Season', y='Average Rainfall', data=seasonal_data, palette='coolwarm')
    plt.title('Average Rainfall by Season across India (1901-2015)', fontsize=16)
    plt.xlabel('Season', fontsize=14)
    plt.ylabel('Average Rainfall (mm)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/seasonal_rainfall_distribution.png')
    plt.close()

# Visualization 5: Heatmap of Correlation between Monthly Rainfall
print("Creating visualization 5: Correlation Heatmap")
# Select only the numeric columns for correlation
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(14, 12))
mask = np.triu(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, mask=mask)
plt.title('Correlation Matrix of Rainfall Data', fontsize=16)
plt.tight_layout()
plt.savefig('plots/rainfall_correlation_heatmap.png')
plt.close()

# Step 3: Data Preprocessing
print("\nStep 3: Data preprocessing")

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

# Visualization 6: Feature Importance Analysis (using top 10 features)
print("Creating visualization 6: Feature Importance Scatter Plot")
plt.figure(figsize=(12, 8))
for col in X.columns[:5]:  # Using first 5 columns for demonstration
    if col in df.columns:
        plt.scatter(df[col], df['ANNUAL'], alpha=0.5, label=col)
plt.title('Relationship Between Selected Features and Annual Rainfall', fontsize=16)
plt.xlabel('Feature Value', fontsize=14)
plt.ylabel('Annual Rainfall (mm)', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('plots/feature_rainfall_relationship.png')
plt.close()

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

# Visualization 7: Actual vs Predicted Rainfall
print("Creating visualization 7: Actual vs Predicted Rainfall")
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Actual vs Predicted Annual Rainfall', fontsize=16)
plt.xlabel('Actual Rainfall (mm)', fontsize=14)
plt.ylabel('Predicted Rainfall (mm)', fontsize=14)
plt.tight_layout()
plt.savefig('plots/actual_vs_predicted.png')
plt.close()

# Visualization 8: Feature Importance from Decision Tree
print("Creating visualization 8: Feature Importance Plot")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance for Rainfall Prediction', fontsize=16)
plt.xlabel('Importance Score', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
plt.close()

# Visualization 9: Residual Plot
print("Creating visualization 9: Residual Plot")
residuals = y_test - y_pred
plt.figure(figsize=(12, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot', fontsize=16)
plt.xlabel('Predicted Values', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.tight_layout()
plt.savefig('plots/residual_plot.png')
plt.close()

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

print("\nVisualizations saved to plots/ directory:")
