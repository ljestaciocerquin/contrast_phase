from pathlib import Path
from guidereg.datasets.loader import load_dataset
from guidereg.propagation.propagate_segmentation import propagate_segmentation


def run_propagation_pipeline(
    config,
    logger=None
):
    dataframe   = load_dataset(config["dataset"]["input_csv"])
    columns     = config["columns"]
    output_root = Path(config["output"]["directory"])
    stages      = config["propagation"]["stages"]

    for _, row in dataframe.iterrows():

        subject_id  = row[columns["subject_id"]]
        scan_date   = row[columns["scan_date"]]
        tumor_path  = row[columns["moving_tumor_mask"]]

        case_directory = (
            output_root
            / str(subject_id)
            / str(scan_date)
        )

        # ==================================
        # BUILD TRANSFORM CHAIN
        # ==================================
        transform_paths = []

        for stage in stages:

            transform_path = (
                case_directory
                / stage
                / "TransformParameters.0.txt"
            )
            transform_paths.append(transform_path)

        # ==================================
        # OUTPUT
        # ==================================
        output_path = (
            case_directory
            / str(stages[-1])
            / "propagated_tumor.nii"
        )

        # ==================================
        # PROPAGATE
        # ==================================
        propagate_segmentation(

            segmentation_path=
            tumor_path,

            transform_paths=
            transform_paths,

            output_path=
            output_path
        )

        if logger:

            logger.info(
                f"Propagated tumor: "
                f"{subject_id}"
            )