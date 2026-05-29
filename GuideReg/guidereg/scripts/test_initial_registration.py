from pathlib import Path
from guidereg.utils.logger import setup_logger
from guidereg.core.initial_registration import run_initial_registration
from guidereg.core.transforms import apply_transform_to_segmentation
from guidereg.core.io import (
    write_image,
    write_segmentation
)

FIXED_IMAGE  = "data/test/fixed_mask.nii.gz"
MOVING_IMAGE = "data/test/moving_mask.nii.gz"
FIXED_MASK   = "data/test/fixed_mask.nii.gz"
MOVING_MASK  = "data/test/moving_mask.nii.gz"
OUTPUT_DIR   = "outputs/test_initial_registration"
FIXED_MASK_LABEL = 5 # Liver

def main():

    logger = setup_logger()
    logger.info(
        "Starting test registration"
    )

    # ======================================
    # REGISTRATION
    # ======================================
    result_image, result_transform = (
        run_initial_registration(
            fixed_image_path    =   FIXED_IMAGE,
            moving_image_path   =   MOVING_IMAGE,
            fixed_mask_path     =   FIXED_MASK,
            fixed_mask_label    =   FIXED_MASK_LABEL,
            output_directory    =   OUTPUT_DIR,
            logger              =   logger
        )
    )

    logger.info(
        "Registration finished"
    )

    # ======================================
    # APPLY TRANSFORM
    # ======================================
    aligned_mask = (
        apply_transform_to_segmentation(
            segmentation_path           =   MOVING_MASK,
            transform_parameter_object  =   result_transform
        )
    )
    logger.info(
        "Transform applied"
    )

    # ======================================
    # SAVE OUTPUTS
    # ======================================
    write_image(
        result_image,
        Path(OUTPUT_DIR)
        / "registered_image.nii.gz"
    )

    write_segmentation(
        aligned_mask,
        Path(OUTPUT_DIR)
        / "aligned_mask.nii.gz"
    )
    logger.info(
        "Outputs saved"
    )


if __name__ == "__main__":
    main()

    #/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/NKI-d23231-00-0193_20080227_kalina99_0644.nii.gz
    #/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/NKI-d23231-00-0193_20080227_kalina99_0644.organs.nii.gz
    #/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/NKI-d23231-00-0193_20070726_kalina99_0642.nii.gz
    #/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/NKI-d23231-00-0193_20070726_kalina99_0642.organs.nii.gz

