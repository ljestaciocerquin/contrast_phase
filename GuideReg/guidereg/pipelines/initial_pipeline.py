from pathlib import Path
from guidereg.core.initial_registration import run_initial_registration
from guidereg.core.transforms import apply_transform_to_segmentation
from guidereg.preprocessing.masks import dilate_mask
from guidereg.core.io import (
    write_image,
    write_segmentation
)
from guidereg.core.io import (
    read_image,
    read_segmentation
)


def run_initial_pipeline(
    case,
    config,
    logger
):
    if logger:
        logger.info(
            f"Processing case: "
            f"{case.subject_id}"
        )

    fixed_image  = read_image(case.fixed_image)
    moving_image = read_image(case.moving_image)
    fixed_mask   = None
    moving_mask  = None

    if case.fixed_mask is not None:
        fixed_mask = read_segmentation(case.fixed_mask, labels=case.fixed_label)
        if fixed_mask is not None and config["masks"]["use_dilation"]:
            fixed_mask = dilate_mask(fixed_mask, radius=config["masks"]["dilation_radius"])

    if case.moving_mask is not None:
        moving_mask_ = moving_mask = read_segmentation(case.moving_mask, labels=case.moving_label)
        if moving_mask is not None and config["masks"]["use_dilation"]:
            moving_mask = dilate_mask(moving_mask, radius=config["masks"]["dilation_radius"])

    # ======================================
    # REGISTRATION
    # ======================================
    result_image, result_transform = (
        run_initial_registration(
            fixed_image      = fixed_image,
            moving_image     = moving_image,
            fixed_mask       = fixed_mask,
            #moving_mask      = moving_mask,
            output_directory = case.output_directory,
            logger           = logger
        )
    )

    # ======================================
    # TRANSFORM MOVING MASK
    # ======================================
    aligned_mask = (
        apply_transform_to_segmentation(
            segmentation                =   moving_mask_,
            transform_parameter_object  =   result_transform
        )
    )

    # ======================================
    # SAVE OUTPUTS
    # ======================================
    output_directory = Path(case.output_directory)

    if config["saving"]["save_registered_images"]:
        write_image(
            result_image,
            output_directory
            / "registered_image.nii.gz"
        )

        write_segmentation(
            aligned_mask,
            output_directory
            / "aligned_mask.nii.gz"
        )

    if logger:
        logger.info(
            f"Finished case: "
            f"{case.subject_id}"
        )