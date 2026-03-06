#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply basic data cleaning,
exporting the result to a new artifact.
"""

import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args: argparse.Namespace) -> None:
    """
    Download a dataset artifact from W&B, clean it, and log a new cleaned artifact.
    """
    # Initialize W&B run
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(vars(args))

    # Download raw artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    logger.info(f"Downloading raw artifact from W&B: {args.input_artifact}")

    df = pd.read_csv(artifact_local_path)

    # Drop outliers
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"])

    # Save cleaned dataset
    output_file = args.output_artifact
    df.to_csv(output_file, index=False)
    logger.info(f"Saved cleaned data to {output_file}")

    # Log the cleaned dataset to W&B
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_file)
    run.log_artifact(artifact)
    run.finish()

    logger.info(f"Uploaded artifact {args.output_artifact} to W&B")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic data cleaning step")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully qualified name of the input artifact to download.",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the cleaned output artifact to create.",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the artifact being created (e.g., cleaned_data).",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the cleaned dataset artifact.",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum listing price to keep in the dataset.",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum listing price to keep in the dataset.",
        required=True,
    )

    args = parser.parse_args()
    go(args)
