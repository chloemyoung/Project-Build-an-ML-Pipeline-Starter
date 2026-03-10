import os
import argparse
import pandas as pd
import mlflow

def go(args):
    # Load CSV from input artifact
    df = pd.read_csv(args.input_artifact)

    # Basic cleaning: filter by price
    df = df[(df["price"] >= args.min_price) & (df["price"] <= args.max_price)]

    df = df.drop_duplicates()

    # Save cleaned CSV
    output_csv_path = os.path.abspath(args.output_artifact)
    df.to_csv(output_csv_path, index=False)
    print(f"Cleaned data saved to: {output_csv_path}")

    # Log artifact to MLflow/W&B
    mlflow.log_artifact(output_csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)
    parser.add_argument("--output_type", type=str, required=False, default="clean_data")  # added
    parser.add_argument("--output_description", type=str, required=False,
                        default="Basic cleaned Airbnb dataset")  # added
    parser.add_argument("--min_price", type=float, required=True)
    parser.add_argument("--max_price", type=float, required=True)
    args = parser.parse_args()
    go(args)
