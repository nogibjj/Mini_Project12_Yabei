[![CI](https://github.com/nogibjj/Mini_Project12_Yabei/actions/workflows/cicd.yml/badge.svg)](https://github.com/nogibjj/Mini_Project12_Yabei/actions/workflows/cicd.yml)
# Mini Project 12

## Overview
This project demonstrates the use of a simple machine learning model in conjunction with MLflow for effective project management and metric tracking. The model, a logistic regression classifier, is trained on a standard dataset (e.g., Iris or Wine dataset from scikit-learn), and MLflow is utilized for tracking experiments, parameters, metrics, and model artifacts.

## Requirements
- Python 3.x
- scikit-learn
- MLflow
- Pandas (optional for data handling)

## Installation
1. Clone the repository:
2. Install the required libraries:
```bash
pip install -r requirements.txt
```

## Running the Project
To run the project, execute the main script:
```bash
python main.py
```
This script will train the logistic regression model on the chosen dataset and log the experiments using MLflow.

## MLflow Tracking
MLflow is used to track:
- Model parameters
- Training metrics (e.g., accuracy)
- Model artifacts
2. Open your browser and navigate to `http://localhost:5000`.

## Project Structure
- `model_training.py`: Main script for training the model and logging with MLflow.
- `test_main.py`: Script for testing the model loading.
- `requirements.txt`: List of Python dependencies.

## Using the MLflow Dashboard
On the MLflow dashboard, you can:
- View a list of all experiments and runs.
- Compare different runs.
- Visualize metrics and parameters.
- Download artifacts.



