from pathlib import Path
import itk


class ITKElastixBackend:

    def __init__(
        self,
        parameter_object,
        output_directory,
        logger=None
    ):
        self.parameter_object = parameter_object
        self.output_directory = Path(output_directory)
        self.logger           = logger


    def register(
        self,
        fixed_image,
        moving_image,
        fixed_mask=None,
        moving_mask=None,
        initial_transform=None
    ):
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        if self.logger:
            self.logger.info(
                "Starting ITKElastix registration"
            )

        result_image, result_transform = (
            itk.elastix_registration_method(
                fixed_image,
                moving_image,
                parameter_object = self.parameter_object,
                fixed_mask       = fixed_mask,
                moving_mask      = moving_mask,
                initial_transform_parameter_object = initial_transform,
                log_to_console   = True,
                log_to_file      = True,
                output_directory=str(self.output_directory)
            )
        )

        if self.logger:
            self.logger.info(
                "Registration completed"
            )

        return result_image, result_transform
