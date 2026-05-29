import itk


def load_transform_parameter_object(
    transform_paths
):

    parameter_object = (
        itk.ParameterObject.New()
    )

    for path in transform_paths:

        parameter_object.AddParameterFile(
            str(path)
        )

    return parameter_object