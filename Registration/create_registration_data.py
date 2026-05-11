import os
import pandas as pd
from pathlib import PureWindowsPath
import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")

# dummy comment to resolve merge conflicts on github
LIVER_LABEL = 5
SERVER_ROOT = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET"
NOT_ON_SERVER_ROOT = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/not_on_server"


def image_path(row, server_folder):
    return os.path.join(
        server_folder,
        PureWindowsPath(row["MatchKey"]).name
    )


def organs_path(image_file):
    return image_file.replace(".nii.gz", ".organs.nii.gz")


def lesion_path(image_file):
    return image_file.replace(".nii.gz", ".lesions.nii.gz")


def create_paired_data(dataset):
    dataset = dataset.copy()

    # Assign server folder
    dataset["server_folder"] = dataset["exist_on_server"].apply(
        lambda x: SERVER_ROOT
        if pd.notna(x)
        else NOT_ON_SERVER_ROOT
    )

    # Filter valid rows
    dataset = dataset[
        (dataset.contrast.isin(["Arterial", "Portal"])) &
        (dataset.is_liver_imaged.isin(["Yes", "Partially"])) &
        (dataset.phase_timing != "0.0")
    ].copy()


    # Keep only exams with both phases
    required_phases = {"Arterial", "Portal"}

    paired_data = (
        dataset
        .sort_values(["SubjectKeyRadiology", "ExamDate"])
        .groupby(["SubjectKeyRadiology", "ExamDate", "server_folder"])
        .filter(lambda g: required_phases.issubset(set(g["contrast"])))
    )
    
    return paired_data


def build_pairs(paired_data):

    pairs = []

    # Per exam
    for (patient_id, exam_date, server_folder), group in paired_data.groupby(
        ["SubjectKeyRadiology", "ExamDate", "server_folder"]
    ):

        arterial = group[group["contrast"] == "Arterial"]
        portal = group[group["contrast"] == "Portal"]

        if arterial.empty or portal.empty:
            continue

        # For each arterial, pair with ALL future portal variants
        for _, a_row in arterial.iterrows():

            a_file = image_path(a_row, server_folder)
            a_organs = organs_path(a_file)
            a_lesions = lesion_path(a_file)

            # Preserve multiple kernel reconstructions
            for _, p_row in portal.iterrows():

                p_file = image_path(p_row, server_folder)
                p_organs = organs_path(p_file)
                p_lesions = lesion_path(p_file)

                pairs.append({
                    "SubjectKeyRadiology": patient_id,
                    "ExamDate": exam_date,

                    # Elastix image inputs
                    "fixed_image": a_file,
                    "moving_image": p_file,

                    # These are TotalSegmentator multi-label masks. elastix_op.py
                    # reads these files and extracts the binary liver mask as
                    # label 5 with load_liver_mask(..., liver_index=5).
                    "fixed_organs": a_organs,
                    "moving_organs": p_organs,
                    "fixed_liver_seg": a_organs,
                    "moving_liver_seg": p_organs,
                    "liver_label": LIVER_LABEL,

                    # Rigid registration config reads these columns. Keep them
                    # present even when lesion masks are unavailable.
                    "fixed_lesion_seg": a_lesions if os.path.exists(a_lesions) else "",
                    "moving_lesion_seg": p_lesions if os.path.exists(p_lesions) else "",

                })

    return pd.DataFrame(pairs).reset_index(drop=True)


def files_load(data_dir, sample=None):

    dataset = pd.read_csv(data_dir)

    paired_data = create_paired_data(dataset)
    
    # -------- build pairs --------
    pairs_df = build_pairs(paired_data)

    # -------- optional sampling --------
    if sample:
        pairs_df = pairs_df.sample(n=sample, random_state=42)


    return pairs_df


def main():
    data_dir = "/projects/net_contrast_classification/contrast_phase/data/cleaned_data_1.csv"

    paired_data = files_load(data_dir)
    save_root = "/projects/net_contrast_classification/contrast_phase/Registration/registration_pairs.csv"
    paired_data.to_csv(save_root, index=False) 


if __name__ == "__main__":
    main()
