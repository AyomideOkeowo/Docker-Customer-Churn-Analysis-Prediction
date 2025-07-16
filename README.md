# Customer-Churn-Analysis-Prediction

This project focuses on predicting customer churn using various machine learning models. The goal was to identify patterns in customer behaviour that indicate a likelihood of leaving a service. After thorough analysis, a Random Forest Classifier was selected as the best performing model and deployed using FastAPI, containerised via Docker for production use.

# Tools & Libraries 
 
 Python Libraries
FastAPI – High-performance API framework for model deployment

Uvicorn – ASGI server for running FastAPI apps

Scikit-learn – Model building and evaluation

Imbalanced-learn (SMOTEENN) – Handling class imbalance

Pandas – Data handling

NumPy – Numerical operations

Matplotlib & Seaborn – Data visualisation

 Deployment Tools
Docker – Containerisation of the FastAPI application

 
 # Project Overview
1. Exploratory Data Analysis (EDA)
Conducted comprehensive EDA to identify trends, correlations, and potential predictors of churn.

Visualised class imbalance and feature distributions using Seaborn and Matplotlib.

2. Data Preprocessing & Feature Engineering
Label Encoding used for categorical variables.

StandardScaler applied to scale numerical features.

Feature selection based on correlation matrix and domain relevance.

3. Handling Class Imbalance
Applied SMOTEENN (Synthetic Minority Oversampling + Edited Nearest Neighbours) to balance the dataset.

4. Model Building & Evaluation
Trained and compared several classification models:

Logistic Regression

Random Forest Classifier

Gradient Boosting Classifier

K-Nearest Neighbours (KNN)

Random Forest Classifier achieved the highest accuracy and robustness.

Evaluated performance using:

Confusion Matrix

Accuracy Score

Classification Report (Precision, Recall, F1-Score)

5. API Development with FastAPI
Created an interactive RESTful API using FastAPI.

Integrated the best-performing model to accept input features via JSON and return predictions.

Used Uvicorn for local and production serving.

6. Containerisation with Docker
Packaged the entire FastAPI application using Docker.

Created a Dockerfile for reproducible builds.

Enabled easy deployment across systems with consistent environments.
