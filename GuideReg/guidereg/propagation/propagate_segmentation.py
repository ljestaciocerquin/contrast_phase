from pathlib import Path
from guidereg.transforms.load_transform import load_transform_parameter_object
from guidereg.core.transforms import apply_transform_to_segmentation
from guidereg.core.io import (
    read_segmentation,
    write_segmentation
)


def propagate_segmentation(
    segmentation_path,
    transform_paths,
    output_path,
    labels=None
):

    # ======================================
    # LOAD SEGMENTATION
    # ======================================
    segmentation = read_segmentation(
        segmentation_path,
        labels=labels
    )

    # ======================================
    # LOAD TRANSFORM CHAIN
    # ======================================
    transform_parameter_object = (
        load_transform_parameter_object(
            transform_paths
        )
    )

    # ======================================
    # APPLY TRANSFORM
    # ======================================
    propagated = (
        apply_transform_to_segmentation(
            segmentation,
            transform_parameter_object
        )
    )

    # ======================================
    # SAVE
    # ======================================
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    write_segmentation(
        propagated,
        output_path
    )