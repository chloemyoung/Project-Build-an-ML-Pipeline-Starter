import wandb


def log_artifact(file_path, artifact_name, artifact_type, artifact_description):
    run = wandb.init()

    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=artifact_description
    )

    artifact.add_file(file_path)
    run.log_artifact(artifact)

    run.finish()
