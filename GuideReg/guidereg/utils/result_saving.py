from pathlib import Path
from guidereg.core.transforms import apply_transform_to_segmentation

from guidereg.core.io import (
    write_image,
    write_segmentation
)

def save_stage_result(
    result,
    moving_mask,
    output_directory,
    save_registered_mask=True
):

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # ======================================
    # SAVE REGISTERED IMAGE
    # ======================================
    '''if save_registered_images:

        write_image(

            result.image,

            output_directory
            / "registered_image.nii.gz"
        )'''

    # ======================================
    # SAVE TRANSFORMED MASK
    # ======================================
    if moving_mask is not None and save_registered_mask:

        aligned_mask = (
            apply_transform_to_segmentation(
                segmentation                =   moving_mask,
                transform_parameter_object  =   result.transform
            )
        )
        write_segmentation(aligned_mask, output_directory/ "result_seg.0.nii")