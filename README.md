# Wind Turbine Power Prediction – ML Pipeline with Flask Deployment
This project demonstrates the development of an end-to-end machine learning pipeline for predicting wind turbine power output based on wind speed, theoretical power, wind direction, and time-based features.

It includes:

### Data ingestion & feature engineering

### Data preprocessing & transformation

### Model training with hyperparameter tuning

### Model evaluation

### Flask-based deployment for real-time prediction

The project integrates best practices for reproducibility and modular design, ensuring maintainability and easy experimentation.

Key Features of the Project

1️⃣ Data Ingestion
Reads raw dataset from CSV

Parses and cleans Date/Time column

Generates time-based features: Month, Day, Year, Hour, Week

Maps months to season categories

Splits dataset into train & test sets

Saves raw, train, and test data in artifacts/

2️⃣ Data Transformation
Separates numerical and categorical features

Handles missing values using SimpleImputer

Scales numerical features using StandardScaler

Encodes categorical features using OneHotEncoder

Saves preprocessing pipeline as proprocessor.pkl for reuse

3️⃣ Model Training
Trains multiple regression models:

Random Forest

Decision Tree

Gradient Boosting

Linear Regression

XGBoost

CatBoost

AdaBoost

Performs RandomizedSearchCV for hyperparameter optimization

Selects best-performing model based on cross-validation score

Saves best model as model.pkl

4️⃣ Model Evaluation
Evaluates the model using R² Score

Ensures chosen model achieves acceptable performance (≥0.6 R² threshold)

5️⃣ Deployment
Implements a Flask API for real-time predictions

Loads preprocessing pipeline and trained model

Accepts input features via web form or API request

Returns predicted wind turbine power

### Goals
Reproducibility: Fixed train-test splits, saved preprocessing objects, and consistent pipelines

Automation: Modular code structure for ingestion, transformation, and training

Deployment Ready: Flask app for quick integration into applications

Performance: Uses hyperparameter tuning to optimize results

### Use Cases
Wind Farm Operators – Predict power output for planning & grid integration

Energy Analysts – Study turbine efficiency and seasonal variations

Research – Test regression models for renewable energy forecasting

### Technology Stack
Python – Core language

Pandas, NumPy – Data processing

Scikit-learn – Pipelines, preprocessing, regression models

CatBoost, XGBoost – Gradient boosting models

Flask – Model deployment

Joblib/Pickle – Saving models & preprocessing objects

