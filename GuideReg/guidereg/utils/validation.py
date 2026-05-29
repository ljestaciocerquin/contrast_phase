from pathlib import Path


def validate_config(
    config: dict,
    task: str
):

    # ======================================
    # TASK-SPECIFIC REQUIRED KEYS
    # ======================================
    if task == "pipeline":

        required_keys = [

            "dataset",
            "columns",
            "output",
            "registration"
        ]

    elif task == "propagate":

        required_keys = [
            "dataset",
            "columns",
            "output",
            "propagation"
        ]

    else:

        raise ValueError(
            f"Unknown task: {task}"
        )

    # ======================================
    # CHECK REQUIRED SECTIONS
    # ======================================
    for key in required_keys:
        if key not in config:
            raise ValueError(

                f"Missing required "
                f"configuration section: "
                f"'{key}'"
            )

    # ======================================
    # CHECK INPUT CSV
    # ======================================
    input_csv = Path(
        config["dataset"]["input_csv"]
    )

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: "
            f"{input_csv}"
        )

    # ======================================
    # REGISTRATION VALIDATION
    # ======================================
    if task == "pipeline":
        valid_stages = [
            "initial",
            "rigid"
        ]

        stages = config["registration"]["stages"]

        for stage in stages:
            if stage not in valid_stages:
                raise ValueError(
                    f"Invalid registration "
                    f"stage: '{stage}'"
                )

    # ======================================
    # PROPAGATION VALIDATION
    # ======================================
    elif task == "propagate":
        valid_stages = [
            "initial",
            "rigid"
        ]

        stages = config["propagation"]["stages"]

        for stage in stages:
            if stage not in valid_stages:
                raise ValueError(
                    f"Invalid propagation "
                    f"stage: '{stage}'"
                )