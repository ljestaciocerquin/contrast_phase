import os
import pandas as pd
import numpy as np
from pathlib import PureWindowsPath
import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")
from Preprocessing.Contrast_data.create_data import group_stratified_train_val_test_split



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

    # Convert time once
    dataset["AcquisitionTime_sec"] = pd.to_timedelta(
        dataset["AcquisitionTime"]
    ).dt.total_seconds()

    # Keep only exams with both phases
    required_phases = {"Arterial", "Portal"}

    paired_data = (
        dataset
        .sort_values(["SubjectKeyRadiology", "ExamDate", "AcquisitionTime_sec"])
        .groupby(["SubjectKeyRadiology", "ExamDate"])
        .filter(lambda g: required_phases.issubset(set(g["contrast"])))
    )
    
    return paired_data

def build_pairs(paired_data):

    pairs = []

    # Per exam
    for (patient_id, exam_date), group in paired_data.groupby(
        ["SubjectKeyRadiology", "ExamDate"]
    ):

        group = group.sort_values("AcquisitionTime_sec")

        arterial = group[group["contrast"] == "Arterial"]
        portal = group[group["contrast"] == "Portal"]

        if arterial.empty or portal.empty:
            continue

        # For each arterial, pair with ALL future portal variants
        for _, a_row in arterial.iterrows():

            a_time = a_row["AcquisitionTime_sec"]

            future_portals = portal[portal["AcquisitionTime_sec"] > a_time]

            if future_portals.empty:
                continue

            a_file = os.path.join(
                a_row["server_folder"],
                PureWindowsPath(a_row["MatchKey"]).name
            )
            a_phase = a_row["phase_timing"]

            # Preserve multiple kernel reconstructions
            for _, p_row in future_portals.iterrows():

                p_time = p_row["AcquisitionTime_sec"]
                interval = p_time - a_time

                if interval <= 0:
                    continue

                p_file = os.path.join(
                    p_row["server_folder"],
                    PureWindowsPath(p_row["MatchKey"]).name
                )

                p_phase = p_row["phase_timing"]

                pairs.append({
                    "SubjectKeyRadiology": patient_id,
                    "ExamDate": exam_date,

                    "arterial_file": a_file,
                    "portal_file": p_file,

                    "arterial_organs": a_file.replace(".nii.gz", ".organs.nii.gz"),
                    "portal_organs": p_file.replace(".nii.gz", ".organs.nii.gz"),


                    "arterial_timing": a_phase,
                    "portal_timing": p_phase,

                    # SAME arterial anchor
                    "time_interval": interval
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

def load_and_split(data_dir):
    paired_data = files_load(data_dir)

    train_idx, val_idx, test_idx = group_stratified_train_val_test_split(
                                                                    np.arange(len(paired_data)),
                                                                    labels=pd.to_datetime(paired_data["ExamDate"]).dt.year,
                                                                    groups=paired_data["SubjectKeyRadiology"]
                                                                    )
    paired_data = paired_data.reset_index(drop=True)

    paired_data["split"] = "unassigned"

    paired_data.loc[train_idx, "split"] = "train"
    paired_data.loc[val_idx, "split"] = "val"
    paired_data.loc[test_idx, "split"] = "test"

    return paired_data


def filter_existing(dataset):
    dataset["exists"] = dataset["output_path"].apply(lambda x: os.path.exists(x))
    filtered_dataset = dataset[~dataset["exists"]].drop(columns=["exists"]).reset_index(drop=True)

    return filtered_dataset
        


def final_dataset(dataset, output_root):

    # -------- output dir --------
    dataset["output_dir"] = dataset["split"].apply(
        lambda x: os.path.join(output_root, x)
    )

    # -------- clean date --------
    dates = pd.to_datetime(dataset["ExamDate"])
    dates_clean = dates.dt.date.astype(str)

    # -------- base key --------
    dataset["base_name_raw"] = (
        dataset["SubjectKeyRadiology"].astype(str)
        + "_"
        + dates_clean
    )

    # -------- count duplicates --------
    dataset["dups_count"] = dataset.groupby("base_name_raw").cumcount()

    # -------- suffix only if dup_id >= 1 --------
    def make_name(row):
        if row["dups_count"] == 0:
            return row["base_name_raw"]
        else:
            return f"{row['base_name_raw']}_{row['dups_count']}"

    dataset["base_name"] = dataset.apply(make_name, axis=1)

    # -------- output path --------
    dataset["output_path"] = dataset.apply(
        lambda row: os.path.join(row["output_dir"], f"{row['base_name']}.pt"),
        axis=1
    )

    return dataset



def main():
    # -------- Load paired data --------
    data_dir = "/projects/net_contrast_classification/contrast_phase/data/cleaned_data/cleaned_data.csv"

    paired_data = load_and_split(data_dir)

    output_root = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/contrast_phase/pairs_preprocessed"
    save_root = "/projects/net_contrast_classification/contrast_phase/Preprocessing/Time_interval_data/paired_preprocessed_data.csv"

    paired_data = final_dataset(paired_data, output_root)

    filtered_data = filter_existing(paired_data) 

    if len(filtered_data) < len(paired_data):
        filtered_data.to_csv("/projects/net_contrast_classification/contrast_phase/Preprocessing/Time_interval_data/missing_paired_preprocessed_data.csv")
    
    paired_data.to_csv(save_root, index=False) 




if __name__ == "__main__":
    main()

