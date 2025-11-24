# src/component_runner.py
import argparse
import os
import sys

# make sure src is importable
repo_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, repo_root)

from src import pipeline_components as pc  # your existing module

def run_data_extraction(args):
    # args.dvc_file, args.output
    out = pc.data_extraction(dvc_file=args.dvc_file, out_path=args.output)
    # if function returns path, optionally copy; but it already writes in your impl
    print("data_extraction done, out:", out)

def run_preprocess(args):
    pc.preprocess(input_csv=args.input, train_out=args.train_out, test_out=args.test_out,
                  test_size=args.test_size, random_state=args.random_state)
    print("preprocess done")

def run_train_model(args):
    out = pc.train_model(train_csv=args.train_csv, model_out=args.model_out,
                         n_estimators=args.n_estimators, random_state=args.random_state)
    print("train_model done, model:", out)

def run_evaluate(args):
    out = pc.evaluate_model(test_csv=args.test_csv, model_in=args.model_in, metrics_out=args.metrics_out)
    print("evaluate done, metrics:", out)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p = subparsers.add_parser("data_extraction")
    p.add_argument("--dvc_file", required=True)
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("preprocess")
    p.add_argument("--input", required=True)
    p.add_argument("--train_out", required=True)
    p.add_argument("--test_out", required=True)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)

    p = subparsers.add_parser("train_model")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--model_out", required=True)
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--random_state", type=int, default=42)

    p = subparsers.add_parser("evaluate")
    p.add_argument("--test_csv", required=True)
    p.add_argument("--model_in", required=True)
    p.add_argument("--metrics_out", required=True)

    args = parser.parse_args()

    if args.cmd == "data_extraction":
        run_data_extraction(args)
    elif args.cmd == "preprocess":
        run_preprocess(args)
    elif args.cmd == "train_model":
        run_train_model(args)
    elif args.cmd == "evaluate":
        run_evaluate(args)

if __name__ == "__main__":
    main()
