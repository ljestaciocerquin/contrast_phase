from pathlib import Path
from guidereg.core.registration_case import RegistrationCase


def build_registration_cases(
    dataframe,
    config
):
    columns     = config["columns"]
    output_root = Path(config["output"]["directory"])
    cases       = []

    for _, row in dataframe.iterrows():

        subject_id = row[columns["subject_id"]]
        scan_date  = row[columns["scan_date"]]

        output_directory = (
            output_root
            / str(subject_id)
            / str(scan_date)
        )

        case = RegistrationCase(
            subject_id  = subject_id,
            fixed_image = row[columns["fixed_image"]],
            moving_image= row[columns["moving_image"]],

            # ==================================
            # OPTIONAL MASKS
            # ==================================
            fixed_mask  = row.get(columns.get("fixed_organ_mask")),
            moving_mask = row.get(columns.get("moving_organ_mask")),

            # ==================================
            # OPTIONAL LABELS
            # ==================================
            fixed_label  = config["registration"].get("fixed_label"),
            moving_label = config["registration"].get("moving_label"),

            # ==================================
            # OPTIONAL TUMOR MASKS
            # ==================================
            fixed_tumor_mask  = row.get(columns.get("fixed_tumor_mask")),
            moving_tumor_mask = row.get(columns.get("moving_tumor_mask")),

            output_directory  = str(output_directory)
        )

        cases.append(case)

    return cases