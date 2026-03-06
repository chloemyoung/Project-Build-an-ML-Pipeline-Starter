import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
]

@hydra.main(version_base=None, config_name="config", config_path=".")
def go(config: DictConfig):

    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    with tempfile.TemporaryDirectory():

        if "download" in active_steps:

            _ = mlflow.run(
                uri="./src/get_data",
                entry_point="main",
                env_manager="local",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw dataset"
                },
            )

        if "basic_cleaning" in active_steps:

            _ = mlflow.run(
                os.path.join("src", "basic_cleaning"),
                entry_point="main",
                env_manager="local",
                parameters={
                    "input_artifact": config["etl"]["sample"],
                    "output_artifact": config["etl"]["output_artifact"],
                    "output_type": config["etl"]["output_type"],
                    "output_description": config["etl"]["output_description"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_split" in active_steps:

            _ = mlflow.run(
                os.path.join("src", "train_val_test_split"),
                entry_point="main",
                env_manager="local",
                parameters={
                    "input": f'{config["etl"]["output_artifact"]}:latest',
                    "test_size": 0.2,
                    "random_seed": 42,
                    "stratify_by": "neighbourhood_group",
                },
            )

        if "train_random_forest" in active_steps:

            rf_config = os.path.abspath("rf_config.json")

            with open(rf_config, "w") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)

            _ = mlflow.run(
                os.path.join("src", "train_random_forest"),
                entry_point="main",
                env_manager="local",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "test_artifact": "test_data.csv:latest",
                    "rf_config": rf_config,
                },
            )

if __name__ == "__main__":
    go()
