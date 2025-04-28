# Rainfall Prediction using Machine Learning

## Overview
This project focuses on predicting the amount of rainfall based on historical weather data using machine learning models. It integrates a frontend (for user interaction) with a backend (for model prediction and data processing).

## Technologies Used
- Frontend: HTML, CSS, JavaScript
- Backend: Python (Flask framework)
- Machine Learning Models: Linear Regression, Random Forest Regressor
- Libraries: scikit-learn, pandas, numpy

## Backend-Frontend Integration
- The frontend uses Fetch API to send HTTP POST requests to the Flask backend.
- The backend receives input, processes it through the trained model, and sends back a JSON response.
- The frontend parses this response and displays the result.

## Evaluation Metrics
- MSE (Mean Squared Error): Measures average squared difference between predicted and actual values.
- RMSE (Root Mean Squared Error): Square root of MSE; easier to interpret because it’s in the same units as the target variable.
- R² Score (Coefficient of Determination): Represents how well the model explains the variability of the target variable.

## Future Improvements
- Add model selection options (choose between Linear Regression and Random Forest from frontend).
- Deploy backend and frontend to cloud (AWS/GCP/Render/Heroku).
- Improve user interface for better usability.
- Collect real-time weather data from APIs.

## Screenshots

