import argparse
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
import os

def go(args):
    # Load cleaned dataset
    df = pd.read_csv(args.input)

    stratify_col = df[args.stratify_by] if args.stratify_by.lower() != "none" else None

    # Split into train+val and test
    trainval_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=stratify_col
    )

    # Further split train+val into train and val (using same fraction)
    val_fraction = 0.2  # can make configurable
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_fraction,
        random_state=args.random_seed,
        stratify=trainval_df[args.stratify_by] if args.stratify_by.lower() != "none" else None
    )

    # Save CSVs
    trainval_path = os.path.abspath("trainval_data.csv")
    test_path = os.path.abspath("test_data.csv")
    train_df.to_csv(trainval_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train+Val saved: {trainval_path}")
    print(f"Test saved: {test_path}")

    # Log artifacts to MLflow/W&B
    mlflow.log_artifact(trainval_path)
    mlflow.log_artifact(test_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--test_size", type=float, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--stratify_by", type=str, required=True)
    args = parser.parse_args()
    go(args)
