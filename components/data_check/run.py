# components/get_data/run.py
import argparse
import pandas as pd
import wandb

def go(args):
    # Start a W&B run
    run = wandb.init(job_type="get_data")
    
    # Log input parameters
    run.config.update(args)
    
    # Download the sample CSV from local path or URL
    df = pd.read_csv(args.sample)
    
    # Save it locally with the artifact name
    df.to_csv(args.artifact_name, index=False)
    
    # Create and log artifact to W&B
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file(args.artifact_name)
    run.log_artifact(artifact)
    
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset and create W&B artifact")
    parser.add_argument("--sample", type=str, required=True, help="Local path or URL to sample CSV")
    parser.add_argument("--artifact_name", type=str, required=True, help="Name for W&B artifact")
    parser.add_argument("--artifact_type", type=str, required=True, help="Artifact type")
    parser.add_argument("--artifact_description", type=str, required=True, help="Artifact description (wrap in single quotes if spaces)")
    
    args = parser.parse_args()
    go(args)
