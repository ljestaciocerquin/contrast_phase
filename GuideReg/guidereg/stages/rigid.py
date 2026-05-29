from guidereg.backends.itk_elastix_backend import ITKElastixBackend
from guidereg.core.parameter_maps import build_rigid_parameter_object
from guidereg.core.stage_result import StageResult
from guidereg.preprocessing.intensity import guess_background_value
from guidereg.utils.stage_paths import get_stage_output_directory
from guidereg.utils.config_saving import save_stage_config

def run_rigid_stage(
    fixed_image,
    moving_image,
    previous_result = None,
    fixed_mask      = None,
    moving_mask     = None,
    output_directory= None,
    logger          = None
):
    # ======================================
    # BACKGROUND INTENSITY
    # ======================================
    pixel_value = guess_background_value(fixed_image)
    
    # ======================================
    # BUILD PARAMETERS
    # ======================================
    use_automatic_initialization = previous_result is None
    parameter_object = build_rigid_parameter_object(
        pixel_value,
        use_automatic_initialization
    )

    # ======================================
    # BACKEND
    # ======================================
    stage_output_directory = (
        get_stage_output_directory(
            output_directory,
            "rigid"
        )
    )

    stage_config = {
        "stage": "rigid",
        "use_masks": fixed_mask is not None,
        "use_moving_masks": moving_mask is not None,
        "automatic_initialization": use_automatic_initialization,
        "metric": "AdvancedMattesMutualInformation",
        "optimizer": "AdaptiveStochasticGradientDescent",
        "iterations": 500
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
    # INITIALIZATION
    # ======================================
    initial_transform = None
    if previous_result is not None:
        initial_transform = previous_result.transform

    # ======================================
    # EXECUTE
    # ======================================
    result_image, result_transform = (
        backend.register(
            fixed_image         =   fixed_image,
            moving_image        =   moving_image,
            fixed_mask          =   fixed_mask,
            moving_mask         =   moving_mask,
            initial_transform   =   initial_transform
        )
    )

    # ======================================
    # RETURN
    # ======================================
    return StageResult(
        image       =   result_image,
        transform   =   result_transform,
        fixed_mask  =   fixed_mask,
        moving_mask =   moving_mask,
        metadata={
            "stage": "rigid",
            "output_directory":
                str(stage_output_directory)
        }
    )