import os
import mlflow
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd

# Use file-based MLflow tracking if running in CI
if os.getenv("GITHUB_ACTIONS"):
    mlflow.set_tracking_uri("file:///tmp/mlruns")
else:
    mlflow.set_tracking_uri("http://localhost:5000")  # Local MLflow server for local runs

mlflow.set_experiment("MLOps_Assignment_Experiment")

# Example MLflow run
with mlflow.start_run():
    # Load dataset
    data = load_boston()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    # Log model and metrics
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "random_forest_model")
