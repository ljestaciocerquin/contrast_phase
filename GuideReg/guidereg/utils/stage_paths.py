from pathlib import Path


def get_stage_output_directory(
    base_output_directory,
    stage_name
):

    stage_directory = (
        Path(base_output_directory)
        / stage_name
    )

    stage_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    return stage_directory