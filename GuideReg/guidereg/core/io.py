import itk
import numpy as np
from pathlib import Path




# ==========================================
# READERS
# ==========================================
def read_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    return itk.imread(
        str(image_path),
        itk.F
    )


def read_segmentation(
    segmentation_path,
    labels=None
):
    segmentation_path = Path(segmentation_path)

    if not segmentation_path.exists():
        raise FileNotFoundError(
            f"Segmentation not found: "
            f"{segmentation_path}"
        )

    segmentation = itk.imread(
        str(segmentation_path),
        itk.UC
    )

    # ======================================
    # NO LABEL FILTERING
    # ======================================
    if labels is None:
        return segmentation

    # ======================================
    # CONVERT SINGLE LABEL TO LIST
    # ======================================
    if isinstance(labels, int):
        labels = [labels]

    # ======================================
    # EXTRACT LABELS
    # ======================================
    array = itk.array_view_from_image(
        segmentation
    )

    binary_mask = np.isin(
        array,
        labels
    ).astype(np.uint8)

    binary_mask = itk.image_view_from_array(
        binary_mask
    )

    # ======================================
    # COPY SPATIAL METADATA
    # ======================================
    binary_mask.CopyInformation(
        segmentation
    )
    return binary_mask


# ==========================================
# WRITERS
# ==========================================
def write_image(
    image,
    output_path
):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    itk.imwrite(
        image,
        str(output_path)
    )


def write_segmentation(
    segmentation,
    output_path
):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    itk.imwrite(
        segmentation,
        str(output_path)
    )