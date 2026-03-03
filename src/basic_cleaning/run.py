#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact.
"""

import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# DO NOT MODIFY
def go(args: argparse.Namespace) -> None:
    """
    Download a dataset artifact from W&B, clean it, and log a new cleaned artifact.

    Args:
        args (argparse.Namespace): Command line arguments including:
            input_artifact (str): Fully qualified name of the input artifact.
            output_artifact (str): Name of the output artifact to create.
            output_type (str): Type of the output artifact.
            output_description (str): Description of the output artifact.
            min_price (float): Minimum price threshold for filtering.
            max_price (float): Maximum price threshold for filtering.
    """

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    run = wandb.init(project="nyc_airbnb", group="cleaning", save_code=True)

    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df["price"].between(min_price, max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"])

    # Save the cleaned data
    df.to_csv("clean_sample.csv", index=False)

    # Log the new data
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

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
