import itk

def dilate_mask(
    mask,
    radius
):

    dimension = mask.GetImageDimension()

    structuring_element = (
        itk.FlatStructuringElement[
            dimension
        ].Ball(radius)
    )

    dilated = itk.binary_dilate_image_filter(
        mask,
        kernel=structuring_element,
        foreground_value=1
    )

    return dilated