from pathlib import Path
import yaml


def save_stage_config(
    config,
    output_directory,
    filename="stage_config.yaml"
):

    output_directory = Path(output_directory)
    
    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    config_path = (
        output_directory
        / filename
    )

    with open(config_path, "w") as f:

        yaml.safe_dump(
            config,
            f,
            sort_keys=False
        )