# pipeline.py
import kfp
from kfp import dsl
from kfp.components import load_component_from_file

data_extraction_op = load_component_from_file("components/data_extraction.yaml")
preprocess_op = load_component_from_file("components/preprocessing.yaml")
train_op = load_component_from_file("components/training.yaml")
eval_op = load_component_from_file("components/evaluation.yaml")

@dsl.pipeline(
    name="housing-pipeline",
    description="Pipeline: extract -> preprocess -> train -> evaluate"
)
def housing_pipeline(dvc_file: str = "data/raw_data.csv"):
    # extract (produces a CSV path)
    extract = data_extraction_op(dvc_file=dvc_file)
    # preprocess (takes extract output)
    preprocess = preprocess_op(input_csv=extract.outputs["data_out"])
    # train
    train = train_op(train_csv=preprocess.outputs["train_csv"])
    # evaluate
    evaluation = eval_op(test_csv=preprocess.outputs["test_csv"], model_in=train.outputs["model_out"])

if __name__ == "__main__":
    from kfp.v2 import compiler
    compiler.Compiler().compile(housing_pipeline, "pipeline.yaml")
    print("Compiled pipeline.yaml")
