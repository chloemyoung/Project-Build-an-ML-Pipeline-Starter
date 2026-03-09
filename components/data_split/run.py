import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb

def go(args):
    # Start a W&B run
    run = wandb.init(project="nyc_airbnb", job_type="data_split")
    
    # Download the cleaned artifact from WandB
    artifact = run.use_artifact(f"{args.input_artifact}:latest")
    artifact_path = artifact.file()
    
    # Read CSV
    df = pd.read_csv(artifact_path)

    # Split the data
    train, test = train_test_split(df, test_size=args.test_size, random_state=args.random_seed)

    # Save train/test CSVs
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)

    # Log train artifact
    train_artifact = wandb.Artifact(
        name=args.train_artifact,
        type=args.train_type,
        description=args.train_description
    )
    train_artifact.add_file("train.csv")
    run.log_artifact(train_artifact)

    # Log test artifact
    test_artifact = wandb.Artifact(
        name=args.test_artifact,
        type=args.test_type,
        description=args.test_description
    )
    test_artifact.add_file("test.csv")
    run.log_artifact(test_artifact)
    
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train and test sets")
    parser.add_argument("--input_artifact", type=str, help="Name of cleaned input artifact")
    parser.add_argument("--train_artifact", type=str, help="Name of train artifact")
    parser.add_argument("--train_type", type=str, help="Type of train artifact")
    parser.add_argument("--train_description", type=str, help="Description of train artifact")
    parser.add_argument("--test_artifact", type=str, help="Name of test artifact")
    parser.add_argument("--test_type", type=str, help="Type of test artifact")
    parser.add_argument("--test_description", type=str, help="Description of test artifact")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for split")
    
    args = parser.parse_args()
    go(args)
