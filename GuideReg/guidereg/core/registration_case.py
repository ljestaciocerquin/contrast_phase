from dataclasses import dataclass
from typing import Optional


@dataclass
class RegistrationCase:

    # ======================================
    # METADATA
    # ======================================
    subject_id: str

    # ======================================
    # IMAGES
    # ======================================
    fixed_image:  str
    moving_image: str

    # ======================================
    # OPTIONAL ORGAN MASKS
    # ======================================
    fixed_mask:  Optional[str] = None
    moving_mask: Optional[str] = None

    # ======================================
    # OPTIONAL LABELS
    # ======================================
    fixed_label:  Optional[int] = None
    moving_label: Optional[int] = None

    # ======================================
    # OPTIONAL TUMOR MASKS
    # ======================================
    fixed_tumor_mask:  Optional[str] = None
    moving_tumor_mask: Optional[str] = None

    # ======================================
    # OUTPUT
    # ======================================
    output_directory: str = ""