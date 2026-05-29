import itk

def build_initial_parameter_object():
    parameter_object = itk.ParameterObject.New()
    parameter_map    = parameter_object.GetDefaultParameterMap("rigid")
    parameter_map["AutomaticTransformInitialization"]       = ["true"]
    parameter_map["MaximumNumberOfIterations"]              = ["500"]
    parameter_map["AutomaticTransformInitializationMethod"] = ["CenterOfGravity"]
    parameter_map["Metric"]                                 = ["AdvancedMeanSquares"]
    parameter_object.AddParameterMap(parameter_map)
    return parameter_object


def build_rigid_parameter_object(
        pixel_value,
        use_automatic_initialization=True
):
    parameter_object = itk.ParameterObject.New()
    parameter_map    = parameter_object.GetDefaultParameterMap("rigid")
    parameter_map["Transform"]                  = ["EulerTransform"]
    parameter_map["NumberOfResolutions"]        = ["3"]
    parameter_map["MaximumNumberOfIterations"]  = ["500"]
    parameter_map["Optimizer"]                  = ["AdaptiveStochasticGradientDescent"]
    parameter_map["Metric"]                     = ["AdvancedMattesMutualInformation"]
    parameter_map["DefaultPixelValue"]          = [str(pixel_value)]
    parameter_map["ErodeFixedMask"]             = ["false"]
    parameter_map["ErodeMovingMask"]            = ["false"]
    parameter_map["AutomaticTransformInitialization"] = ["true" if use_automatic_initialization else "false"]
    parameter_object.AddParameterMap(parameter_map)
    return parameter_object