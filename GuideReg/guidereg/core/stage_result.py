from dataclasses import dataclass
from typing import Optional


@dataclass
class StageResult:

    # ======================================
    # MAIN OUTPUTS
    # ======================================
    image:     object
    transform: object

    # ======================================
    # OPTIONAL MASKS
    # ======================================
    fixed_mask:  Optional[object] = None
    moving_mask: Optional[object] = None

    # ======================================
    # METADATA
    # ======================================
    metadata: Optional[dict] = None