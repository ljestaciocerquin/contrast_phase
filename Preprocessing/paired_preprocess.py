import os, torch
import pandas as pd
import numpy as np
# from pathlib import Path
from pathlib import PureWindowsPath
from monai.transforms import (Compose, LoadImage, Resized, Spacingd, ScaleIntensityRanged)

# from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from monai.data import MetaTensor
from cnn_preprocess import group_train_test_split

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

    # Convert time once
    dataset["AcquisitionTime_sec"] = pd.to_timedelta(
        dataset["AcquisitionTime"]
    ).dt.total_seconds()

    # Keep only exams with both phases
    required_phases = {"Arterial", "Portal"}

    paired_data = (
        dataset
        .sort_values(["SubjectKeyRadiology", "ExamDate", "AcquisitionTime_sec"])
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

            a_time = a_row["AcquisitionTime_sec"]

            future_portals = portal[portal["AcquisitionTime_sec"] > a_time]

            if future_portals.empty:
                continue

            a_file = os.path.join(
                server_folder,
                PureWindowsPath(a_row["MatchKey"]).name
            )

            # Preserve multiple kernel reconstructions
            for _, p_row in future_portals.iterrows():

                p_time = p_row["AcquisitionTime_sec"]
                interval = p_time - a_time

                if interval <= 0:
                    continue

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

    return (
        pairs_df["SubjectKeyRadiology"].tolist(),
        pairs_df["ExamDate"].tolist(),

        pairs_df["arterial_file"].tolist(),
        pairs_df["portal_file"].tolist(),
        pairs_df["arterial_organs"].tolist(),
        pairs_df["portal_organs"].tolist(),
        pairs_df["time_interval"].tolist()
    )


class PairedPreprocess:
    def __init__(self, patient_ids, dates, a_files, p_files,
            a_organs, p_organs, organ_ids, intervals, test_size = 0.2,
            pixdim=(1,1,1), resize = (128,128,128)
            ):
    
        self.patient_ids=patient_ids
        self.dates = dates
        self.a_files = a_files
        self.p_files = p_files
        self.a_organs = a_organs
        self.p_organs = p_organs
        self.organ_ids=organ_ids
        self.intervals = intervals
        self.test_size=test_size
        self.pixdim=pixdim
        self.resize=resize


        self.loader=LoadImage(image_only=True, ensure_channel_first=True)
        self.transforms = Compose([
                                    Spacingd(
                                        keys=["a_image", "p_image", "a_mask", "p_mask"],
                                        pixdim=self.pixdim,
                                        mode=["bilinear", "bilinear", "nearest", "nearest"],
                                        allow_missing_keys=True,
                                    ),
                                    ScaleIntensityRanged(
                                        keys=["a_image", "p_image"],
                                        a_min=-100, a_max=300,
                                        b_min=0.0, b_max=1.0,
                                        clip=True,
                                    ),
                                    Resized(
                                        keys=["a_image", "p_image", "a_mask", "p_mask"],
                                        spatial_size=self.resize,
                                        mode=["bilinear", "bilinear", "nearest", "nearest"],
                                        allow_missing_keys=True,
                                    ),
])

    def data_split(self, image_files, patient_ids):
        all_indices = np.arange(len(image_files))

        # First split: train+val vs test
        temp_idx, test_idx = group_train_test_split(all_indices, test_size=0.2, groups=patient_ids)

        # Second split: train vs val
        train_idx, val_idx = group_train_test_split(temp_idx, test_size=0.2, groups=np.array(patient_ids)[temp_idx])

        return train_idx, val_idx, test_idx
    
    def safe_load(self, path):
        try:
            if path is None or pd.isna(path):
                return None
            return self.loader(path)
        except Exception as e:
            print(f"Load failed: {path} | {e}")

            return None

    def process_organs(self, mask_raw, affine):
        if mask_raw is None:
            return None

        try:
            if mask_raw.ndim == 3:
                mask_raw = mask_raw.unsqueeze(0)

            if np.size(mask_raw) == 0:
                return None

            organ_array = multi_channel(self.organ_ids, mask_raw)
            organ_array = torch.from_numpy(organ_array.astype(np.float32))

            return MetaTensor(organ_array, affine=affine)

        except Exception as e:
            print(f"Organ processing failed: {e}")
            return None

    def data_load(self, idx):

        # -------- Load images --------
        a_image = self.safe_load(self.a_files[idx])
        p_image = self.safe_load(self.p_files[idx])

        a_affine = a_image.affine if a_image is not None else None
        p_affine = p_image.affine if p_image is not None else None

        # -------- Load raw masks --------
        a_organs_raw = self.safe_load(self.a_organs[idx])
        p_organs_raw = self.safe_load(self.p_organs[idx])

        # -------- Process masks --------
        a_organs = self.process_organs(a_organs_raw, a_affine)
        p_organs = self.process_organs(p_organs_raw, p_affine)

        times = self.intervals[idx]

        return a_image, p_image, a_organs, p_organs, times

    def cropping(self, image, organ_array, lesions=None):
        # Create single-channel foreground mask
        foreground_mask = torch.from_numpy((organ_array.sum(axis=0) > 0).astype(np.float32))

        # Compute bounding box
        mask_nonzero = torch.nonzero(foreground_mask)
        if mask_nonzero.numel() == 0:
            # fallback if mask is empty
            zmin, ymin, xmin = 0, 0, 0
            zmax, ymax, xmax = foreground_mask.shape
        else:
            zmin, ymin, xmin = mask_nonzero.min(0)[0]
            zmax, ymax, xmax = mask_nonzero.max(0)[0] + 1

        # Optional margin
        margin = 20
        zmin, ymin, xmin = max(zmin - margin, 0), max(ymin - margin, 0), max(xmin - margin, 0)
        zmax = min(zmax + margin, foreground_mask.shape[0])
        ymax = min(ymax + margin, foreground_mask.shape[1])
        xmax = min(xmax + margin, foreground_mask.shape[2])

        # Crop image, organ_array
        image = image[:, zmin:zmax, ymin:ymax, xmin:xmax]
        if organ_array is not None:
            organ_array = organ_array[:, zmin:zmax, ymin:ymax, xmin:xmax]
        if lesions is not None:
            lesions = lesions[:, zmin:zmax, ymin:ymax, xmin:xmax]

        return image, organ_array, lesions

    def preprocess_sample(self, idx):

        # -------- Load data --------
        a_image, p_image, a_organs, p_organs, times = self.data_load(idx)

        # Build MONAI dictionary
        data = {
            "a_image": a_image,
            "p_image": p_image,
            "a_mask": a_organs,
            "p_mask": p_organs,
        }

        # -------- Cropping --------
        if a_organs is not None and p_organs is not None:
            try:
                a_image, a_organs, _ = self.cropping(a_image, a_organs)
                p_image, p_organs, _ = self.cropping(p_image, p_organs)

                data["a_image"] = a_image
                data["p_image"] = p_image
                data["a_mask"] = a_organs
                data["p_mask"] = p_organs

            except Exception as e:
                print(f"Cropping failed: {e}", flush=True)

        # -------- MONAI transforms --------
        try:
            data = self.transforms(data)

            print(
                f"Transforms done! "
                f"a_image: {data['a_image'].shape}, "
                f"p_image: {data['p_image'].shape}",
                flush=True
            )

        except Exception as e:
            print(f"Transforms failed: {e}", flush=True)

        return (
            data["a_image"],
            data["p_image"],
            data.get("a_mask"),
            data.get("p_mask"),
            times
        )

    def process_and_save(self, indices, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        counter = 0
        skipped = 0

        for idx in indices:
            try:
                a_img, p_img, a_mask, p_mask, interval = self.preprocess_sample(idx)

                base_name = f"{self.patient_ids[idx]}_{self.dates[idx]}_{idx}"

                samples = [(a_img, p_img, a_mask, p_mask, interval)]

                # -------- Save --------
                for i, (a_i, p_i, a_m, p_m, t) in enumerate(samples):

                    save_path = os.path.join(save_dir, f"{base_name}.pt")

                    torch.save({
                        "patient_id": self.patient_ids[idx],
                        "exam_date": self.dates[idx],

                        "arterial_image": a_i,
                        "portal_image": p_i,

                        # "arterial_mask": a_m,
                        # "portal_mask": p_m,

                        "time_interval": t
                    }, save_path)

                    counter += 1

            except Exception as e:
                print(f"Skipping {idx}: {e}", flush=True)
                skipped += 1

        return counter, skipped

    def run(self, output_root, split):

        train_idx, val_idx, test_idx = self.data_split(
            self.a_files, self.patient_ids   # use paired indexing source
        )

        print("Split done!\n", flush=True)

        if split == "train":
            count, skipped = self.process_and_save(
                train_idx, os.path.join(output_root), "train"
            )
            print(f"\nTrain done! Saved {count}, skipped {skipped}", flush=True)

        elif split == "val":
            count, skipped = self.process_and_save(
                val_idx, os.path.join(output_root), "val"
            )
            print(f"\nValidation done! Saved {count}, skipped {skipped}", flush=True)

        elif split == "test":
            count, skipped = self.process_and_save(
                test_idx, os.path.join(output_root), "test"
            )
            print(f"\nTest done! Saved {count}, skipped {skipped}", flush=True)

def main():
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    print(f"Running task {task_id}", flush=True)
    
    # -------- Load paired data --------
    data_dir = "/projects/net_contrast_classification/contrast_phase/data/cleaned_data_1.csv"

    patient_ids, dates, a_files, p_files, a_organs, p_organs, intervals = files_load(
        data_dir, sample=None
    )

    organ_ids = [1,2,3,5,8,9,13,51,52,63,64,65,66]

    preprocessor = PairedPreprocess(
        patient_ids,
        dates,
        a_files,
        p_files,
        a_organs,
        p_organs,
        organ_ids,
        intervals,
        test_size=0.2,
        pixdim=(1,1,1),
        resize=(128,128,128)
    )

    # -------- Split ONCE (pair-level) --------
    train_idx, val_idx, test_idx = preprocessor.data_split(
        a_files,   # <-- use paired index reference
        patient_ids
    )

    split_map = {
        0: ("train", train_idx),
        1: ("val", val_idx),
        2: ("test", test_idx),
    }

    # -------- Assign split --------
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