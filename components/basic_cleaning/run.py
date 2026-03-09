import argparse
import wandb
import pandas as pd

def main(args):
    run = wandb.init(project="nyc_airbnb", job_type="basic_cleaning")
    
    # Download the artifact
    artifact = run.use_artifact(args.input_artifact)
    df = pd.read_csv(artifact.file())

    # Basic cleaning
    df = df[df["price"].between(args.min_price, args.max_price)]
    
    # Save cleaned CSV
    df.to_csv(args.output_artifact, index=False)

    # Log to W&B
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)
    parser.add_argument("--output_type", type=str, required=True)
    parser.add_argument("--output_description", type=str, required=True)
    parser.add_argument("--min_price", type=float, required=True)
    parser.add_argument("--max_price", type=float, required=True)
    args = parser.parse_args()
    main(args)
