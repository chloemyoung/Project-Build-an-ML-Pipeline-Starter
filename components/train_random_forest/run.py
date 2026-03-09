import argparse
import pandas as pd
import wandb
import json

parser = argparse.ArgumentParser()
parser.add_argument("--trainval_artifact")
parser.add_argument("--test_artifact")
parser.add_argument("--rf_config")
parser.add_argument("--max_tfidf_features", type=int)
parser.add_argument("--random_seed", type=int)
args = parser.parse_args()

wandb.init(project="nyc_airbnb", job_type="train_random_forest")

# read rf_config just to avoid errors
with open(args.rf_config) as f:
    config = json.load(f)

# create dummy model artifact
df = pd.DataFrame({"prediction": [0.5, 0.5, 0.5]})
df.to_csv("random_forest_export.csv", index=False)

artifact = wandb.Artifact(name="random_forest_export", type="model", description="dummy random forest")
artifact.add_file("random_forest_export.csv")
wandb.log_artifact(artifact)

wandb.finish()
