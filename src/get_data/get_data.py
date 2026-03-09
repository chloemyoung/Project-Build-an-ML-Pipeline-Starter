import os
import argparse
import pandas as pd
import wandb

def go(args):
    run = wandb.init(job_type="download_data")

    # Get project root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # CSV input and output paths
    csv_path = os.path.join(PROJECT_ROOT, "data", "sample1.csv")
    sample_csv_path = os.path.join(PROJECT_ROOT, "sample.csv")

    # --- DEBUGGING: print paths ---
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"CSV input path: {csv_path}")
    print(f"CSV output path: {sample_csv_path}")
    print(f"CSV exists? {os.path.exists(csv_path)}")

    # Stop execution here if file not found
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find CSV file at {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Sample fraction
    df = df.sample(frac=args.sample)

    # Save locally before pushing to W&B
    df.to_csv(sample_csv_path, index=False)

    # Create W&B artifact
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description
    )
    artifact.add_file(sample_csv_path)

    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, required=True)
    parser.add_argument("--artifact_name", type=str, required=True)
    parser.add_argument("--artifact_type", type=str, required=True)
    parser.add_argument("--artifact_description", type=str, required=True)
    args = parser.parse_args()
    go(args)
