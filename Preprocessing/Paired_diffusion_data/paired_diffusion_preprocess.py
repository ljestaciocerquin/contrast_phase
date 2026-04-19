import os, torch
import pandas as pd
import numpy as np
# from pathlib import Path
from pathlib import PureWindowsPath
from monai.transforms import (Compose, LoadImage, Resized, Spacingd, ScaleIntensityRanged)

# from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from monai.data import MetaTensor
from contrast_phase.Preprocessing.Contrast_data.cnn_preprocess import group_train_test_split

import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")

from Radiomics.radiomics_pipeline import multi_channel
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import GroupShuffleSplit
from itertools import product


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
        (dataset.is_liver_imaged == "Yes") &
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

                    "arterial_file": a_file,
                    "portal_file": p_file,

                    "arterial_organs": a_file.replace(".nii.gz", ".organs.nii.gz"),
                    "portal_organs": p_file.replace(".nii.gz", ".organs.nii.gz"),

                    "arterial_seg": a_file.replace(".nii.gz", ".seg.nii.gz"),
                    "portal_seg": p_file.replace(".nii.gz", ".seg.nii.gz")
                })

    return pd.DataFrame(pairs).reset_index(drop=True)


def files_load(data_dir, sample=None):

    dataset = pd.read_csv(data_dir)

    dataset['server_folder'] = dataset.exist_on_server.apply(
        lambda x: "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET"
        if pd.notna(x)
        else "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/not_on_server"
    )

    dataset = dataset[
        (dataset.contrast.isin(['Arterial', "Portal"])) &
        (dataset.is_liver_imaged == "Yes") &
        (dataset.phase_timing != '0.0')
    ]

    dataset = dataset.sort_values(
        by=["SubjectKeyRadiology", "ExamDate", "contrast"]
    )

    # -------- build pairs --------
    pairs_df = build_pairs(dataset)

    # -------- optional sampling --------
    if sample:
        pairs_df = pairs_df.sample(n=sample, random_state=42)

    return (
        pairs_df["SubjectKeyRadiology"].tolist(),
        pairs_df["ExamDate"],
        pairs_df["arterial_file"].tolist(),
        pairs_df["portal_file"].tolist(),
        pairs_df["arterial_organs"].tolist(),
        pairs_df["portal_organs"].tolist(),
        pairs_df["arterial_seg"].tolist(),
        pairs_df["portal_seg"].tolist()
    )

def group_train_test_split(indices, test_size=0.2, groups=None):
    """
    Splits indices into train/test without splitting the same group across sets.
    
    Args:
        indices: array-like of indices to split (e.g., np.arange(len(images)))
        test_size: fraction of data to use as test
        groups: array-like of same length as indices indicating group membership
    
    Returns:
        train_idx, test_idx: arrays of indices
    """
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(indices, groups=groups))

    return train_idx, test_idx


class PairedDiffusionPreprocess:
    def __init__(self, patient_ids, dates, a_files, p_files,
            a_organs, p_organs, a_seg, p_seg, organ_ids, intervals, test_size = 0.2,
            pixdim=(1,1,1), resize = (128,128,128)
            ):
    
        self.patient_ids=patient_ids
        self.dates = dates
        self.a_files = a_files
        self.p_files = p_files
        self.a_organs = a_organs
        self.p_organs = p_organs
        self.a_organs = a_seg
        self.p_organs = p_seg
        self.organ_ids=organ_ids
        self.test_size=test_size
        self.pixdim=pixdim
        self.resize=resize


        self.loader=LoadImage(image_only=True, ensure_channel_first=True)
        self.transforms = Compose([
                                    Spacingd(
                                        keys=["a_image", "p_image", "a_mask", "p_mask", "a_seg", "p_seg"],
                                        pixdim=self.pixdim,
                                        mode=["bilinear", "bilinear", "nearest", "nearest", "nearest", "nearest"],
                                        allow_missing_keys=True,
                                    ),
                                    ScaleIntensityRanged(
                                        keys=["a_image", "p_image"],
                                        a_min=-100, a_max=300,
                                        b_min=0.0, b_max=1.0,
                                        clip=True,
                                    ),
                                    Resized(
                                        keys=["a_image", "p_image", "a_mask", "p_mask", "a_seg", "p_seg"],
                                        spatial_size=self.resize,
                                        mode=["bilinear", "bilinear", "nearest", "nearest", "nearest", "nearest"],
                                        allow_missing_keys=True,
                                    )
                                       ])
        
        # self.register = # ADD IMAGE REGISTRATION

    def data_split(self, image_files, patient_ids):
        all_indices = np.arange(len(image_files))

        # First split: train+val vs test
        temp_idx, test_idx = group_train_test_split(all_indices, test_size=0.2, groups=patient_ids)

        # Second split: train vs val
        train_idx, val_idx = group_train_test_split(temp_idx, test_size=0.2, groups=np.array(patient_ids)[temp_idx])

        return train_idx, val_idx, test_idx
    

    def preprocess_sample(self,idx):
        return

    def preprocess_and_save():
        return
    
    def run(self, output_root, split):
        return



def main():
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    print(f"Running task {task_id}", flush=True)
    
    # -------- Load paired data --------
    data_dir = "/projects/net_contrast_classification/contrast_phase/data/cleaned_data_1.csv"

    patient_ids, dates, a_files, p_files, a_organs, p_organs, a_seg, p_seg = files_load(
        data_dir, sample=None
    )

    image_files = [f"{pid}_{date}" for pid, date in zip(patient_ids, dates)]

    organ_ids = [5]

    preprocessor = PairedDiffusionPreprocess(
        patient_ids,
        dates,
        a_files,
        p_files,
        a_organs,
        p_organs,
        a_seg,
        p_seg,
        organ_ids,
        test_size=0.2,
        pixdim=(1,1,1),
        resize=(128,128,128)
    )


    # -------- Shared split paths --------
    split_dir = "/projects/net_contrast_classification/contrast_phase/Preprocessing/paired_diffusion_splits"
    os.makedirs(split_dir, exist_ok=True)

    train_path = os.path.join(split_dir, "train_idx.npy")
    val_path   = os.path.join(split_dir, "val_idx.npy")
    test_path  = os.path.join(split_dir, "test_idx.npy")


    # -------- Create or load split --------
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        print("Creating split...", flush=True)

        train_idx, val_idx, test_idx = preprocessor.data_split(image_files, patient_ids)

        np.save(train_path, train_idx)
        np.save(val_path, val_idx)
        np.save(test_path, test_idx)

    else:
        print("Loading existing split...", flush=True)
        train_idx = np.load(train_path)
        val_idx   = np.load(val_path)
        test_idx  = np.load(test_path)

    
    # -------- Assign split --------

    split_map = {
        0: ("train", train_idx),
        1: ("val", val_idx),
        2: ("test", test_idx),
    }

    split_id = task_id % 3
    split_name, indices = split_map[split_id]

    # -------- Chunking --------
    chunks_per_split = 10
    chunk_id = task_id // 3

    chunks = np.array_split(indices, chunks_per_split)

    if chunk_id >= len(chunks):
        print("Nothing to process for this task.", flush=True)
        return

    my_indices = chunks[chunk_id]

    print(
        f"Processing {split_name} | chunk {chunk_id} | size {len(my_indices)}",
        flush=True
    )

    # -------- Output --------
    output_dir = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/pairs_preprocessed"

    preprocessor.process_and_save(
        my_indices,
        os.path.join(output_dir, split_name)
    )




if __name__ == "__main__":
    main()

