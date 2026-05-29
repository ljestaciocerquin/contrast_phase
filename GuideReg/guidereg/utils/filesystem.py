from pathlib import Path


def create_directory(directory):

    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True
    )
    return directory