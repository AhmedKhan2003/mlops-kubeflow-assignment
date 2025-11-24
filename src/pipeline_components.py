# src/pipeline_components.py
import os
import subprocess
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# NOTE: these functions are plain Python functions. We'll compile them
# to Kubeflow components using kfp.components.create_component_from_func in compile_components.py

def data_extraction(dvc_file="data/raw_data.csv", out_path="/tmp/data/raw_data.csv") -> str:
    """
    Pull the data tracked by DVC to the local workspace and return local path.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Pull from remote (assumes dvc remote configured)
    subprocess.check_call(["dvc", "pull", dvc_file], cwd=os.getcwd())
    # Copy to out_path (file already in data/)
    src = os.path.join(os.getcwd(), dvc_file)
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} not found after dvc pull")
    # ensure the same path structure
    return src

def preprocess(input_csv: str, train_out: str="/tmp/data/train.csv", test_out: str="/tmp/data/test.csv", test_size: float=0.2, random_state:int=42) -> (str, str):
    """
    Read CSV, create a classification target (binary) from MEDV if present,
    scale features, and write train/test CSVs. Returns (train_out, test_out).
    """
    df = pd.read_csv(input_csv)
    # If target exists as 'MEDV' or 'target', try MEDV for Boston
    if 'MEDV' in df.columns:
        target_col = 'MEDV'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        # if no target, assume last column is target
        target_col = df.columns[-1]

    # Convert regression target to binary classification: above median -> 1
    df['label'] = (df[target_col] > df[target_col].median()).astype(int)

    X = df.drop(columns=[target_col, 'label']) if target_col in df.columns else df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # scale numeric features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    train_df = pd.DataFrame(X_train_s, columns=X.columns)
    train_df['label'] = y_train.values
    test_df = pd.DataFrame(X_test_s, columns=X.columns)
    test_df['label'] = y_test.values

    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    # Save scaler in same dir (optional)
    joblib.dump(scaler, os.path.join(os.path.dirname(train_out), "scaler.joblib"))

    return train_out, test_out

def train_model(train_csv: str, model_out: str="/tmp/model/rf_model.joblib", n_estimators:int=100, random_state:int=42) -> str:
    """
    Train a RandomForest classifier and save the model artifact.
    """
    df = pd.read_csv(train_csv)
    X = df.drop(columns=['label'])
    y = df['label']
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X, y)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(clf, model_out)
    return model_out

def evaluate_model(test_csv: str, model_in: str, metrics_out: str="/tmp/results/metrics.txt") -> str:
    """
    Load model, evaluate on test set, and write simple metrics to metrics_out.
    """
    df = pd.read_csv(test_csv)
    X = df.drop(columns=['label'])
    y = df['label']
    clf = joblib.load(model_in)
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    with open(metrics_out, "w") as fh:
        fh.write(f"accuracy: {acc}\n")
        fh.write(f"f1_score: {f1}\n")
    return metrics_out
