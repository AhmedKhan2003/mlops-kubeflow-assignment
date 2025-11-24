# src/compile_components.py
from kfp import components
from pipeline_components import data_extraction, preprocess, train_model, evaluate_model

components.create_component_from_func(
    data_extraction, base_image="python:3.10-slim", output_component_file="../components/data_extraction.yaml"
)
components.create_component_from_func(
    preprocess, base_image="python:3.10-slim", output_component_file="../components/preprocessing.yaml"
)
components.create_component_from_func(
    train_model, base_image="python:3.10-slim", output_component_file="../components/training.yaml"
)
components.create_component_from_func(
    evaluate_model, base_image="python:3.10-slim", output_component_file="../components/evaluation.yaml"
)
print("Components written to components/*.yaml")
