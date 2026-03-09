# main.py
import mlflow
from pathlib import Path
from omegaconf import OmegaConf

# Load config
config_path = Path("config/config.yaml")
config = OmegaConf.load(config_path)

# Root project path
project_root = Path(__file__).parent

# Check which steps to run (Hydra-compatible)
steps = config.main.get("steps", [])

# Example: if you want to allow passing via CLI with Hydra:
# python main.py +main.steps=[get_data,basic_cleaning,train_model]
if isinstance(steps, str):
    steps = [steps]  # ensure list

# --- Step 1: Get Data ---
if "get_data" in steps or "all" in steps:
    mlflow.run(
        str(project_root / "components/get_data"),
        parameters={
            "sample": "sample1.csv",
            "artifact_name": "sample.csv",
            "artifact_type": "raw_data",
            "artifact_description": "'Raw_file_as_downloaded'",  # single quotes to avoid split
        },
        experiment_name="nyc_airbnb",
    )

# --- Step 2: Basic Cleaning ---
if "basic_cleaning" in steps or "all" in steps:
    mlflow.run(
        str(project_root / "components/basic_cleaning"),
        parameters={
            "input_artifact": "sample.csv:latest",  # W&B collection:alias format
            "output_artifact": "clean_sample.csv",
            "output_type": "clean_data",
            "output_description": "'Cleaned_Airbnb_data'",  # wrap in single quotes
            "min_price": config.main.min_price,
            "max_price": config.main.max_price,
        },
        experiment_name="nyc_airbnb",
    )

# --- Step 3: Training Model ---
if "train_model" in steps or "all" in steps:
    mlflow.run(
        str(project_root / "components/train_model"),
        parameters={
            "input_artifact": "clean_sample.csv:latest",
            "test_size": config.main.test_size,
            "random_seed": config.main.random_seed,
            "output_model": "model.pkl",
        },
        experiment_name="nyc_airbnb",
    )
