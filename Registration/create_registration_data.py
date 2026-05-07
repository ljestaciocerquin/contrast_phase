import os
import pandas as pd
from pathlib import PureWindowsPath
import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")

def create_paired_data(dataset):
    dataset = dataset.copy()

    # Assign server folder
    dataset["server_folder"] = dataset["exist_on_server"].apply(
        lambda x: "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET"
        if pd.notna(x)
        else "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/not_on_server"
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

        group = group.sort_values("AcquisitionTime_sec")

        arterial = group[group["contrast"] == "Arterial"]
        portal = group[group["contrast"] == "Portal"]

        if arterial.empty or portal.empty:
            continue

        # For each arterial, pair with ALL future portal variants
        for _, a_row in arterial.iterrows():

            a_file = os.path.join(
                server_folder,
                PureWindowsPath(a_row["MatchKey"]).name
            )

            # Preserve multiple kernel reconstructions
            for _, p_row in portal.iterrows():

                p_file = os.path.join(
                    server_folder,
                    PureWindowsPath(p_row["MatchKey"]).name
                )

                pairs.append({
                    "SubjectKeyRadiology": patient_id,
                    "ExamDate": exam_date,

                    "fixed_image": a_file,
                    "moving_image": p_file,

                    "fixed_organs": a_file.replace(".nii.gz", ".organs.nii.gz"),
                    "moving_organs": p_file.replace(".nii.gz", ".organs.nii.gz"),

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