# src/pipeline_components.py

import os
import subprocess
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from kfp.dsl import component
from kfp.components import InputPath, OutputPath

@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib", "dvc"]
)
def data_extraction(dvc_file: str, data_out_path: OutputPath(str)):
    """
    Pull data via DVC and write to local CSV.
    """
    # ensure output folder
    os.makedirs(os.path.dirname(data_out_path), exist_ok=True)
    # pull from remote
    subprocess.check_call(["dvc", "pull", dvc_file], cwd=os.getcwd())
    # after pull, file should exist locally
    local_path = os.path.join(os.getcwd(), dvc_file)
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"DVC file not found at {local_path}")
    # read and save
    df = pd.read_csv(local_path)
    df.to_csv(data_out_path, index=False)

@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib", "numpy"]
)
def preprocess(input_csv: InputPath(str),
               train_csv: OutputPath(str),
               test_csv: OutputPath(str),
               test_size: float = 0.2,
               random_state: int = 42):
    """
    Preprocess: read CSV, create binary label, split, scale, save train/test.
    """
    df = pd.read_csv(input_csv)
    target_col = 'MEDV' if 'MEDV' in df.columns else df.columns[-1]
    df['label'] = (df[target_col] > df[target_col].median()).astype(int)

    features = df.drop(columns=[target_col, 'label']) if target_col in df.columns else df.drop(columns=['label'])
    labels = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    train_df = pd.DataFrame(X_train_s, columns=features.columns)
    train_df['label'] = y_train.values
    test_df = pd.DataFrame(X_test_s, columns=features.columns)
    test_df['label'] = y_test.values

    os.makedirs(os.path.dirname(train_csv), exist_ok=True)
    os.makedirs(os.path.dirname(test_csv), exist_ok=True)

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Save scaler
    joblib.dump(scaler, os.path.join(os.path.dirname(train_csv), "scaler.joblib"))

@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def train_model(train_csv: InputPath(str),
                model_out: OutputPath(str),
                n_estimators: int = 100,
                random_state: int = 42):
    """
    Train RandomForest classifier and save the model.
    """
    df = pd.read_csv(train_csv)
    X = df.drop(columns=['label'])
    y = df['label']
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X, y)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(clf, model_out)

@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def evaluate_model(test_csv: InputPath(str),
                   model_in: InputPath(str),
                   metrics_out: OutputPath(str)):
    """
    Load model, evaluate, and write metrics.
    """
    df = pd.read_csv(test_csv)
    X = df.drop(columns=['label'])
    y = df['label']

    clf = joblib.load(model_in)
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)

    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    with open(metrics_out, "w") as f:
        f.write(f"accuracy: {acc}\n")
        f.write(f"f1_score: {f1}\n")
