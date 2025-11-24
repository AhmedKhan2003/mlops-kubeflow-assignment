# pipeline.py
import kfp
from kfp import dsl
from kfp import components

# load compiled components
data_extraction_op = components.load_component_from_file("components/data_extraction.yaml")
preprocess_op = components.load_component_from_file("components/preprocessing.yaml")
train_op = components.load_component_from_file("components/training.yaml")
eval_op = components.load_component_from_file("components/evaluation.yaml")

@dsl.pipeline(
    name="housing-classification-pipeline",
    description="A simple pipeline converting housing prices to a binary classification and training a RF"
)
def housing_pipeline():
    # 1. extract/pull data
    extract = data_extraction_op(dvc_file="data/raw_data.csv")
    # 2. preprocess: use the path returned by extract
    preprocess = preprocess_op(input_csv=extract.output)
    # 3. training
    train = train_op(train_csv=preprocess.outputs['train_out'])
    # 4. evaluation
    evaluation = eval_op(test_csv=preprocess.outputs['test_out'], model_in=train.output)

if __name__ == "__main__":
    import kfp
    from kfp import compiler
    compiler.Compiler().compile(housing_pipeline, __file__.replace(".py", ".yaml"))
    print("Compiled pipeline to pipeline.yaml")
