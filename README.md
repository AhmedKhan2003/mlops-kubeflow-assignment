# 🚀 MLOps Kubeflow Assignment

## 📖 Project Overview
This project implements a **complete Machine Learning Operations (MLOps) pipeline** using:  
- **Kubeflow Pipelines** for orchestration  
- **DVC** for data versioning  
- **MLflow** for experiment tracking  
- **Python & Scikit-learn** for model training  
- **GitHub Actions** for CI/CD automation  

The ML task is a **regression/classification problem** using the **California Housing dataset**.  
The pipeline includes:
1. **Data Extraction**
2. **Data Preprocessing**
3. **Model Training**
4. **Model Evaluation**
5. **Experiment Logging with MLflow**

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone <YOUR_REPO_URL>
cd mlops-kubeflow-assignment
2️⃣ Python Environment
bash


python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
3️⃣ DVC Remote Setup
bash

dvc remote add -d myremote <REMOTE_STORAGE_URL>
dvc pull
4️⃣ Minikube & Kubeflow Pipelines
bash

minikube start
# Install Kubeflow Pipelines (standalone or full)
# Access dashboard:
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

🏗️ Pipeline Walkthrough
Compile Components
bash
python src/pipeline_components.py  # Create component YAMLs
Define Pipeline
bash

python pipeline.py  # Generates pipeline.yaml
Run Pipeline
Upload pipeline.yaml to Kubeflow Pipelines UI

Trigger pipeline execution

Monitor status & logs for each step

🔄 Continuous Integration
The GitHub Actions workflow automates:

Checkout code

Setup Python 3.10

Install dependencies

Run MLflow experiment script

Upload MLflow artifacts

Workflow file: .github/workflows/ci.yml

📂 Artifacts & Outputs
ML models saved in mlruns/ (logged via MLflow)

Metrics for evaluation are also logged in MLflow

DVC ensures versioned dataset integrity

📌 Notes
Make sure Minikube and Kubeflow Pipelines are properly installed before running the pipeline

For MLflow logging, ensure your tracking server is running (default: http://localhost:5000)