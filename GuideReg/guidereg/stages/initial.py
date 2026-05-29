from guidereg.backends.itk_elastix_backend import ITKElastixBackend
from guidereg.core.parameter_maps import build_initial_parameter_object
from guidereg.core.stage_result import StageResult
from guidereg.utils.stage_paths import get_stage_output_directory
from guidereg.utils.config_saving import save_stage_config

def run_initial_stage(
    fixed_image,
    moving_image,
    fixed_mask=None,
    moving_mask=None,
    output_directory=None,
    logger=None
):

    # ======================================
    # BUILD PARAMETERS
    # ======================================
    parameter_object = (
        build_initial_parameter_object()
    )

    # ======================================
    # BACKEND
    # ======================================
    stage_output_directory = (
        get_stage_output_directory(
            output_directory,
            "initial"
        )
    )

    stage_config = {

        "stage": "initial",

        "use_masks":
            fixed_mask is not None,

        "use_moving_masks":
            moving_mask is not None,

        "parameter_map":
            "initial"
    }

    save_stage_config(
        stage_config,
        stage_output_directory
    )

    backend = ITKElastixBackend(
        parameter_object = parameter_object,
        output_directory = stage_output_directory,
        logger           = logger
    )

    # ======================================
    # EXECUTE
    # ======================================
    result_image, result_transform = (
        backend.register(
            fixed_image     =   fixed_image,
            moving_image    =   moving_image,
            fixed_mask      =   fixed_mask,
            moving_mask     =   moving_mask
        )
    )

    # ======================================
    # RETURN STAGE RESULT
    # ======================================
    return StageResult(
        image       =   result_image,
        transform   =   result_transform,
        fixed_mask  =   fixed_mask,
        moving_mask =   moving_mask,
        metadata={
            "stage": "initial",
            "output_directory":
                str(stage_output_directory)
        }
    )