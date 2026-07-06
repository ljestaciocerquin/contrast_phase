import os
import pandas as pd
from pathlib import PureWindowsPath, Path
import sys
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
sys.path.append("/projects/net_contrast_classification/contrast_phase")
from Preprocessing.Contrast_data.create_data import group_stratified_train_val_test_split


def load_complete_registred_pairs(registered_path, server_folder):

    registered_cases = pd.read_csv(registered_path)

    registered_cases = registered_cases.drop(index=1390).reset_index(drop=True) # Faulty sample

    output_path = Path(server_folder)

    registered_cases["registered_image"] = registered_cases.apply(
        lambda row: output_path
        / str(row["SubjectKeyRadiology"])
        / str(row["ExamDate"])
        / "rigid"
        / "result.0.nii",
        axis=1
    )

    registered_cases["registered_liver"] = registered_cases.apply(
        lambda row: output_path
        / str(row["SubjectKeyRadiology"])
        / str(row["ExamDate"])
        / "rigid"
        / "result_seg.0.nii",
        axis=1
    )

    registered_cases["registered_lesion"] = registered_cases.apply(
        lambda row: output_path
        / str(row["SubjectKeyRadiology"])
        / str(row["ExamDate"])
        / "rigid"
        / "propagated_tumor.nii",
        axis=1
    )

    registered_cases = registered_cases.rename(
    columns={
        "fixed_image": "arterial_image",
        # "moving_image": "portal_image_raw",
        "registered_image": "portal_image",
        "fixed_organs": "arterial_organs",
        # "moving_organs": "portal_organs_raw",
        "registered_liver": "portal_organs",
        "fixed_lesion_seg": "arterial_lesion",
        # "moving_lesion_seg": "portal_lesion_raw",
        "registered_lesion": "portal_lesion",
        }
    )

    registered_cases = registered_cases[
        [
            "SubjectKeyRadiology",
            "ExamDate",
            "arterial_image",
            # "portal_image_raw",
            "portal_image",
            "arterial_organs",
            # "portal_organs_raw",
            "portal_organs",
            "arterial_lesion",
            # "portal_lesion_raw",
            "portal_lesion",
            "arterial_lesionfree",
            "portal_lesionfree",
            "lesion_visible_both"
        ]
    ]

    return registered_cases

def create_incomplete_pairs(dataset):
    """
    Identify cases where ONLY one phase (Arterial or Portal)
    is available for a given subject and exam date.
    """

    dataset = dataset.copy()

    dataset["server_folder"] = dataset["exist_on_server"].apply(
        lambda x: "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET"
        if pd.notna(x)
        else "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/not_on_server"
    )

    # filter valid scans
    dataset = dataset[
        (dataset.contrast.isin(["Arterial", "Portal"])) &
        (dataset.is_liver_imaged.isin(["Yes", "Partially"])) &
        (dataset.phase_timing != "0.0")
    ].copy()

    grouped = dataset.groupby(
        ["SubjectKeyRadiology", "ExamDate"]
    )

    incomplete = []

    for (patient_id, exam_date), group in grouped:

        has_A = (group["contrast"] == "Arterial").any()
        has_P = (group["contrast"] == "Portal").any()

        # keep ONLY incomplete exams
        if has_A and has_P:
            continue

        arterial = group[group["contrast"] == "Arterial"]
        portal = group[group["contrast"] == "Portal"]

        # build same-style records as complete pairs
        for _, a_row in arterial.iterrows():

            a_file = os.path.join(
                a_row["server_folder"],
                PureWindowsPath(a_row["MatchKey"]).name
            )

            incomplete.append({
                "SubjectKeyRadiology": patient_id,
                "ExamDate": exam_date,

                "arterial_image": a_file,
                "portal_image": pd.NA,

                "arterial_organs": a_file.replace(".nii.gz", ".organs.nii.gz"),
                "portal_organs": pd.NA,
                # ,

                "arterial_lesion": a_file.replace(".nii.gz", ".seg.nii.gz"),
                "portal_lesion": pd.NA,

                "present_phase": "arterial"
            })

        for _, p_row in portal.iterrows():

            p_file = os.path.join(
                p_row["server_folder"],
                PureWindowsPath(p_row["MatchKey"]).name
            )

            incomplete.append({
                "SubjectKeyRadiology": patient_id,
                "ExamDate": exam_date,

                "arterial_image": pd.NA,
                "portal_image": p_file,

                "arterial_organs": pd.NA,
                "portal_organs": p_file.replace(".nii.gz", ".organs.nii.gz"),
                # ,

                "arterial_lesion": pd.NA,
                "portal_lesion": p_file.replace(".nii.gz", ".seg.nii.gz"),

                "present_phase": "portal"
            })

    return pd.DataFrame(incomplete).reset_index(drop=True)


def load_and_split(data_dir, registered_path, server_folder):
    dataset = pd.read_csv(data_dir)

    complete_df = load_complete_registred_pairs(registered_path, server_folder)
    complete_df = complete_df.reset_index(drop=True)
    complete_df["split"] = "unassigned"

    lesion_visible_one = complete_df[complete_df["lesion_visible_both"] == False].reset_index(drop=True)
    # lesion_visible_one["split"] = "test"

    lesion_visible_both = complete_df[complete_df["lesion_visible_both"] == True].reset_index(drop=True)
    print(f"Total complete pairs: {len(complete_df)}")
    print(f"Pairs with lesion visible in both phases: {len(lesion_visible_both)}")
    print(f"Pairs with lesion visible in only one phase: {len(lesion_visible_one)}")

    incomplete_df = create_incomplete_pairs(dataset)
    incomplete_df = incomplete_df.reset_index(drop=True)
    incomplete_df["split"] = "inference"


    train_idx, val_idx, test_idx = group_stratified_train_val_test_split(
                                                                    np.arange(len(complete_df)),
                                                                    labels=pd.to_datetime(complete_df["ExamDate"]).dt.year,
                                                                    groups=complete_df["SubjectKeyRadiology"],
                                                                    # # train_size=0.7,
                                                                    # val_size=0.2,
                                                                    # test_size=0.1,
                                                                    )

    complete_df.loc[train_idx, "split"] = "train"
    complete_df.loc[val_idx, "split"] = "val"
    complete_df.loc[test_idx, "split"] = "test"

    # complete_df = (
    #             pd.concat(
    #                 [lesion_visible_both, lesion_visible_one],
    #                 ignore_index=True
    #             )
    #             .sample(frac=1, random_state=42)
    #             .reset_index(drop=True)
    #             ) 
    
    full_df = pd.concat([complete_df, incomplete_df], ignore_index=True)
    full_df = full_df.reset_index(drop=True)

    return full_df


def filter_existing(dataset):
    dataset["exists"] = dataset["output_path"].apply(lambda x: os.path.exists(x))
    filtered_dataset = dataset[~dataset["exists"]].drop(columns=["exists"]).reset_index(drop=True)

    return filtered_dataset
        


def final_dataset(dataset, output_root):

    if not os.path.exists(output_root):
        os.makedirs(output_root)

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
    data_dir = "/projects/net_contrast_classification/contrast_phase/data/cleaned_data/final_data.csv"
    registered_path = "/projects/net_contrast_classification/contrast_phase/GuideReg/guidereg/data/registration_pairs.csv" 
    server_folder = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/contrast_phase/registered_images1"
    
    full_df = load_and_split(data_dir, registered_path, server_folder)
    
    output_server_path = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/contrast_phase/segmentation_data"

    full_df = final_dataset(full_df, output_server_path)

    save_root = ("/projects/net_contrast_classification/contrast_phase/Preprocessing/Segmentation_data")

    full_df.to_csv(save_root + "/full_pairs.csv", index=False)
    # incomplete_df.to_csv(save_root + "/incomplete_pairs.csv", index=False)

    print(full_df.info())


if __name__ == "__main__":
    main()