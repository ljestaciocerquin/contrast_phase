import os
import pandas as pd
import numpy as np
from pathlib import PureWindowsPath

import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import StratifiedGroupKFold



def contrast_timing(dataset):
    dataset["contrast_timing"] = dataset["contrast"]
    mask = dataset["contrast"] != "Non-contrast"
    dataset.loc[mask, "contrast_timing"] = (dataset.loc[mask, "contrast"] + " " + dataset.loc[mask, "phase_timing"].astype(str))
    dataset.loc[mask, "contrast_timing"] = dataset.loc[mask, "contrast_timing"].str.strip()
    dataset.loc[dataset["contrast"] == "Non-contrast", "contrast_timing"] = "Non-contrast"   

    return dataset

def files_load(data_dir, sample=None):
    dataset = pd.read_csv(data_dir)

    dataset['server_folder'] = dataset.exist_on_server.apply(
        lambda x: "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET"
        if pd.notna(x)
        else "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/not_on_server"
    )

    dataset = dataset[
        (dataset.contrast.isin(['Arterial', "Portal", "Non-contrast"])) &
        (dataset.is_liver_imaged.isin(["Yes", "Partially"])) 
    ]

    dataset.loc[dataset["contrast"] == "Non-contrast", "phase_timing"] = None
    dataset = contrast_timing(dataset)

    if sample is not None:
        dataset = (dataset.groupby('contrast_timing', dropna=False, group_keys=False)
                   .apply(lambda x: x.sample(n=min(len(x), sample), random_state=42))
                   .reset_index(drop=True))

        dataset = contrast_timing(dataset)
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # ---------------- BUILD FILE PATHS ----------------
    dataset["image_file"] = dataset.apply(
        lambda row: os.path.join(
            row["server_folder"],
            PureWindowsPath(row["MatchKey"]).name
        ),
        axis=1
    )

    dataset["organ_file"] = dataset["image_file"].str.replace(".nii.gz", ".organs.nii.gz")

    # ---------------- SELECT FINAL COLUMNS ----------------
    keep_cols = [
        "SubjectKeyRadiology",
        "ExamDate",
        "MatchKey",
        "contrast",
        "contrast_timing",
        "image_file",
        "organ_file",
    ]

    dataset = dataset[keep_cols].copy()

    dataset = dataset.rename(columns={
        "contrast_timing": "phase"
    })

    rare_classes = ["Portal Too Late",
                    "Portal Too Early",
                    "Arterial Too Late"]

    dataset["rare_class"] = dataset["phase"].isin(rare_classes).astype(int)

    return dataset

def group_stratified_train_val_test_split(indices, labels, groups, test_size=0.2, val_size=0.2, random_state=42):

    indices = np.array(indices)
    labels = np.array(labels)
    groups = np.array(groups)

    # -------------------------
    # STEP 1: train vs test
    # -------------------------
    n_splits_test = int(1 / test_size)

    sgkf_test = StratifiedGroupKFold(
        n_splits=n_splits_test,
        shuffle=True,
        random_state=random_state
    )

    train_val_idx, test_idx = next(
        sgkf_test.split(indices, y=labels, groups=groups)
    )

    # subset for train/val
    indices_tv = indices[train_val_idx]
    labels_tv = labels[train_val_idx]
    groups_tv = groups[train_val_idx]

    # -------------------------
    # STEP 2: train vs val
    # -------------------------

    val_relative = val_size / (1 - test_size)
    n_splits_val = int(1 / val_relative)

    sgkf_val = StratifiedGroupKFold(
        n_splits=n_splits_val,
        shuffle=True,
        random_state=random_state
    )

    train_idx_rel, val_idx_rel = next(
        sgkf_val.split(indices_tv, y=labels_tv, groups=groups_tv)
    )

    train_idx = indices_tv[train_idx_rel]
    val_idx = indices_tv[val_idx_rel]

    return train_idx, val_idx, test_idx

def load_and_split(data_dir):
    data = files_load(data_dir, sample=None)

    train_idx, val_idx, test_idx = group_stratified_train_val_test_split(
                                                                    np.arange(len(data)),
                                                                    labels=data["phase"],
                                                                    groups=data["SubjectKeyRadiology"]
                                                                    )
    data = data.reset_index(drop=True)

    data["split"] = "unassigned"

    data.loc[train_idx, "split"] = "train"
    data.loc[val_idx, "split"] = "val"
    data.loc[test_idx, "split"] = "test"

    data["augment"] = ((data["split"] == "train") & (data["rare_class"] == 1)).astype(int)

    return data

def save_final_dataset(dataset, output_root, save_root):

    dataset["output_dir"] = dataset["split"].apply(
        lambda x: os.path.join(output_root, x)
    )

    dataset["base_name"] = dataset["MatchKey"].apply(
        lambda x: PureWindowsPath(x).name.replace(".nii.gz", "")
    )

    dataset["output_path"] = dataset.apply(
        lambda row: os.path.join(row["output_dir"], f"{row['base_name']}.pt"),
        axis=1
    )

    dataset.to_csv(save_root)

    return dataset



def main():
    # -------- Load paired data --------
    data_dir = "/projects/net_contrast_classification/contrast_phase/data/cleaned_data/final_data.csv"

    data = load_and_split(data_dir)

    output_root = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/contrast_phase/contrast_preprocessed" 
    save_root = "/projects/net_contrast_classification/contrast_phase/Preprocessing/Contrast_data/preprocessed_data.csv"

    data = save_final_dataset(data, output_root, save_root)

    # filtered_data = save_final_dataset(data) 

    # if len(filtered_data) < len(paired_data):
    #     filtered_data.to_csv("/projects/net_contrast_classification/contrast_phase/Preprocessing/Time_interval_data/missing_paired_preprocessed_data.csv")
    
    data.to_csv(save_root, index=False) 




if __name__ == "__main__":
    main()