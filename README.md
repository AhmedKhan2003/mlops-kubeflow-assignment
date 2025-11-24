# MLOps Kubeflow Assignment

## Student
- Full name: <Your Full Name>
- Student ID: <Your Student ID>

## Project Overview
This project demonstrates a Kubeflow Pipelines based MLOps workflow for a binary classification variant of the Boston housing dataset.

## Structure
(brief file/folder listing)

## Setup Instructions
1. Install dependencies:
pip install -r requirements.txt

kotlin
Copy code
2. Initialize DVC & pull data:
dvc init
dvc remote add -d myremote <remote-url>
dvc pull

markdown
Copy code
3. Start Minikube and install Kubeflow Pipelines (see official KFP docs).
4. Compile components and pipeline:
python src/compile_components.py
python pipeline.py

yaml
Copy code
5. Upload `pipeline.yaml` in KFP UI and run.

## Pipeline Walkthrough
- Step 1: data_extraction — pulls via DVC.
- Step 2: preprocessing — scales features and creates binary label.
- Step 3: training — trains RandomForest classifier and saves model.
- Step 4: evaluation — evaluates model and writes metrics.

## How to run CI
- Jenkins: pipeline reads `Jenkinsfile`.
- GitHub Actions: see `.github/workflows/ci.yml`.

## Notes
- Dataset: Boston housing (converted to binary labels)
- Model artifact format: joblib
