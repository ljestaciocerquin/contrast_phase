import itk


def apply_transform_to_segmentation(
    segmentation,
    transform_parameter_object
):

    # ======================================
    # FORCE NEAREST-NEIGHBOR
    # ACROSS ALL STAGES
    # ======================================
    number_of_maps = (
        transform_parameter_object
        .GetNumberOfParameterMaps()
    )

    for i in range(number_of_maps):

        parameter_map = (
            transform_parameter_object
            .GetParameterMap(i)
        )
        parameter_map["FinalBSplineInterpolationOrder"] = ["0"]

        transform_parameter_object.SetParameterMap(
            i,
            parameter_map
        )

    # ======================================
    # APPLY TRANSFORM
    # ======================================
    result = itk.transformix_filter(

        segmentation,

        transform_parameter_object=
        transform_parameter_object
    )

    return result
