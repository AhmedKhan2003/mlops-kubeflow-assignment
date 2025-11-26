# src/compile_components.py

from kfp.components import create_component_from_func
from pipeline_components import data_extraction, preprocess, train_model, evaluate_model
import os

# Ensure components directory exists
os.makedirs(os.path.join(os.path.dirname(__file__), "../components"), exist_ok=True)

# Create YAML for each component
create_component_from_func(
    func=data_extraction,
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib", "dvc"],
    output_component_file="../components/data_extraction.yaml"
)

create_component_from_func(
    func=preprocess,
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib", "numpy"],
    output_component_file="../components/preprocessing.yaml"
)

create_component_from_func(
    func=train_model,
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib"],
    output_component_file="../components/training.yaml"
)

create_component_from_func(
    func=evaluate_model,
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib"],
    output_component_file="../components/evaluation.yaml"
)

print("Component YAMLs generated in components/ folder.")
