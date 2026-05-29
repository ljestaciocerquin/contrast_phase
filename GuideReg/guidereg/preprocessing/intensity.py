from collections import Counter
import itk
import numpy as np


def guess_background_value(
    image,
    rim=3
):
    array   = itk.array_view_from_image(image)
    z, y, x = array.shape

    edges = np.concatenate([
        array[:rim,:,:].ravel(),
        array[-rim:,:,:].ravel(),
        array[:, :rim,:].ravel(),
        array[:, -rim:,:].ravel(),
        array[:, :, :rim].ravel(),
        array[:, :, -rim:].ravel()
    ])

    return int(
        Counter(
            edges.tolist()
        ).most_common(1)[0][0]
    )