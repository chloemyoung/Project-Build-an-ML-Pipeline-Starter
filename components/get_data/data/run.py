import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import argparse
import pandas as pd
import wandb

def go(args):
    run = wandb.init(project="nyc_airbnb", job_type="basic_cleaning")

    df = pd.read_csv(args.input_artifact)
    df = df.drop_duplicates()
    df = df.dropna()
    df = df[(df['price'] >= args.min_price) & (df['price'] <= args.max_price)]

    clean_path = "clean_data.csv"
    df.to_csv(clean_path, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file(clean_path)
    run.log_artifact(artifact)
    run.finish()
    print("Cleaned artifact logged!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_artifact", required=True)
    parser.add_argument("--output_artifact", required=True)
    parser.add_argument("--output_type", required=True)
    parser.add_argument("--output_description", required=True)
    parser.add_argument("--min_price", type=float, required=True)
    parser.add_argument("--max_price", type=float, required=True)
    args = parser.parse_args()
    go(args)
