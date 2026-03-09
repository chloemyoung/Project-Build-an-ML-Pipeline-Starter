import argparse
import logging
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def go(args):

    run = wandb.init(job_type="download_data")

    logger.info(f"Downloading {args.sample}")

    # Load dataset
    df = pd.read_csv(args.sample)

    # Save locally
    df.to_csv(args.artifact_name, index=False)

    # Create artifact
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )

    artifact.add_file(args.artifact_name)

    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--sample", type=str)
    parser.add_argument("--artifact_name", type=str)
    parser.add_argument("--artifact_type", type=str)
    parser.add_argument("--artifact_description", type=str)

    args = parser.parse_args()

    go(args)
