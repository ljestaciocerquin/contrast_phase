from guidereg.core.io import (
    read_image,
    read_segmentation
)
from guidereg.preprocessing.masks import dilate_mask
from guidereg.stages.initial import run_initial_stage
from guidereg.stages.rigid import run_rigid_stage
from guidereg.utils.result_saving import save_stage_result

def run_registration_pipeline(
    case,
    config,
    logger=None
):
    
    if logger:
        logger.info(
            f"Processing case: "
            f"{case.subject_id}"
        )

    # ======================================
    # LOAD IMAGES
    # ======================================
    fixed_image  = read_image(case.fixed_image)
    moving_image = read_image(case.moving_image)

    # ======================================
    # OPTIONAL MASKS
    # ======================================
    fixed_mask  = None
    moving_mask = None

    if case.fixed_mask is not None:
        fixed_mask = read_segmentation(
            case.fixed_mask,
            labels=case.fixed_label
        )

    if case.moving_mask is not None:
        moving_mask = read_segmentation(
            case.moving_mask,
            labels=case.moving_label
        )

    # ======================================
    # OPTIONAL DILATION
    # ======================================
    if (fixed_mask is not None and config["masks"]["use_dilation"]):
        fixed_dilated_mask = dilate_mask(
            fixed_mask,
            radius = config["masks"]["dilation_radius"]
        )

    if (moving_mask is not None and config["masks"]["use_dilation"]):
        moving_dilated_mask = dilate_mask(
            moving_mask,
            radius = config["masks"]["dilation_radius"]
        )

    # ======================================
    # STAGE EXECUTION
    # ======================================
    result = None
    for stage_name in config["registration"]["stages"]:
        if stage_name == "initial":
            result = run_initial_stage(
                fixed_image         = fixed_mask, #fixed_image,
                moving_image        = moving_mask, #moving_image,
                fixed_mask          = fixed_dilated_mask,
                #moving_mask         = moving_mask_dil,
                output_directory    = case.output_directory,
                logger              = logger
            )
            save_stage_result(
                result                  =   result,
                moving_mask             =   moving_mask,
                output_directory        =   result.metadata["output_directory"],
                save_registered_mask    =   config["saving"]["save_initial_registered_mask"]
            )
            
        elif stage_name == "rigid":
            result = run_rigid_stage(
                fixed_image     =   fixed_image,
                moving_image    =   moving_image,
                previous_result =   result,
                fixed_mask      =   fixed_dilated_mask,
                moving_mask     =   moving_dilated_mask,
                output_directory=   case.output_directory,
                logger          =   logger
            )
            save_stage_result(
                result                  =   result,
                moving_mask             =   moving_mask,
                output_directory        =   result.metadata["output_directory"],
                save_registered_mask    =   config["saving"]["save_rigid_registered_mask"]
            )

        else:
            raise ValueError(
                f"Unknown stage: "
                f"{stage_name}"
            )

    #return result