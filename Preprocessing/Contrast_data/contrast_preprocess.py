
import os, torch
import pandas as pd
import numpy as np
# from pathlib import Path
from pathlib import PureWindowsPath
from monai.transforms import (Compose, LoadImage, Resized, Spacingd
                              , CropForegroundd, ScaleIntensityRange
                              , Resize, RandFlipd, RandRotated, RandGaussianNoised,
                              RandScaleIntensityd, RandAffined)

# from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from monai.data import MetaTensor

import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")

from Radiomics.radiomics_pipeline import multi_channel
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold



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
        (dataset.is_liver_imaged.isin(["Yes", "Partially"])) &
        (dataset.phase_timing != '0.0')
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


class Preprocess3D:
    def __init__(self, dataset, organ_ids, pixdim=(1,1,1), resize = (128,128,128), augment = False):
        
        self.dataset=dataset
        self.organ_ids=organ_ids
        self.pixdim=pixdim
        self.resize=resize
        self.augment=augment
        self.loader=LoadImage(image_only=True, ensure_channel_first=True)
        self.spacing = Spacingd(
                                    keys=["image", "mask"],
                                    pixdim=(1,1,1),
                                    mode=["bilinear", "nearest"],
                                    allow_missing_keys=True)                # allow missing lesions key
        

        self.intensity_scaler = ScaleIntensityRange(a_min=-100, a_max=300, b_min=0.0, b_max=1.0, clip=True)


        self.resizer = Resized(
                            keys=["image", "mask"],
                            spatial_size=self.resize,
                            mode=["bilinear", "nearest"],
                            allow_missing_keys=True)                 # allow missing lesions key


        self.augment_transform = Compose([
                                        RandRotated(keys=["image", "mask"], 
                                                    range_x = [0.1, 0.1],                   # ~ 5-6 degrees
                                                    range_y = [0.1, 0.1],
                                                    range_z = [0.1, 0.1], 
                                                    prob=0.8,
                                                    padding_mode='reflection',              # mirror filling in the empty space after rotation
                                                    allow_missing_keys=True),      

                                        RandGaussianNoised(keys=["image"], prob=0.2,
                                                            mean=0.0, std=0.01,
                                                            allow_missing_keys=True),

                                        RandScaleIntensityd(keys=["image"], factors=0.01, prob=0.2, allow_missing_keys=True),

                                        RandAffined(keys=["image", "mask"],                 # random affine with translation and scaling
                                                    mode=["bilinear", "nearest"],
                                                    padding_mode='reflection',
                                                    translate_range=(3,3,3),                # randomly select pixel/voxel to translate for every spatial dims
                                                    scale_range=(0.05,0.05,0.05),           # randomly select the scale factor to translate for every spatial dims
                                                    shear_range=(0.05,0.05,0.05),
                                                    prob=0.3,
                                                    allow_missing_keys=True
                                                )
                                    ])

    def data_load(self, idx):
        row = self.dataset.iloc[idx]

        image = self.loader(row["image_file"])
        image_affine = image.affine
        organ_array = None

        contrast = row["contrast"]
        phase = row["phase"]

        if self.organ_ids is not None:
            try:
                organ_array_raw = self.loader(row["organ_file"])

                if organ_array_raw.ndim == 3:
                    organ_array_raw = organ_array_raw.unsqueeze(0)

                organ_array = multi_channel(self.organ_ids, organ_array_raw)
                organ_array = torch.from_numpy(organ_array.astype(np.float32))
                organ_array = MetaTensor(organ_array, affine=image_affine)

            except Exception as e:
                print(f"Fallback: {e}")

        return image, organ_array, contrast, phase

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
    
    def augmentation(self, image, organ_array, idx, num_aug=10):
        row = self.dataset.iloc[idx]

        augmented_samples = [(image, organ_array)]

        if row["augment"] == 1:
            for _ in range(num_aug):
                data = {"image": image.clone(), "mask": organ_array.clone()}
                data = self.augment_transform(data)
                augmented_samples.append((data["image"], data["mask"]))
        else:
            if np.random.rand() < 0.2:
                data = {"image": image.clone(), "mask": organ_array.clone()}
                data = self.augment_transform(data)
                augmented_samples.append((data["image"], data["mask"]))

        return augmented_samples
    
    def preprocess_sample(self, idx):

        # ---------- Load data -----------
        image, organ_array, contrast, phase = self.data_load(idx)

        # -------- Crop using organ mask --------
        if organ_array is not None:
            image, organ_array, _ = self.cropping(image, organ_array)
            try:
                print(f"Cropping done! Image: {image.shape}, Organ array: {organ_array.shape}", flush=True)
            except Exception as e:
                print(f"Cropping failed: {e}", flush=True)

        
        # -------- Resampling --------

        data = {"image": image, "mask": organ_array}
        data = self.spacing(data)
        image = data["image"]
        organ_array = data["mask"]
        try:
            print(f"Spacing done! Image shape:{image.shape}, Organ array shape: {organ_array.shape}", flush=True)
        except Exception as e:
            print(f"Spacing failed: {e}", flush=True)

        
        # -------- Intensity scaling --------
    
        image = self.intensity_scaler(image)
        try:
            print(f"Intensity scaling done! Image shape:{image.shape}", flush=True)
        except Exception as e:
            print(f"Intensity scaling failed: {e}", flush=True)


        # -------- Resizing --------

        data = {"image": image, "mask": organ_array}
        data = self.resizer(data)
        image = data["image"]
        organ_array = data["mask"]

        try:
            print(f"Resizing done! Image shape:{image.shape}, Organ array shape: {organ_array.shape}", flush=True)
        except Exception as e:
            print(f"Resizing failed: {e}", flush=True)


        return image, organ_array, contrast, phase

    def process_and_save(self, indices):
        counter = 0
        skipped = 0

        for idx in indices:
            row = self.dataset.iloc[idx]

            try:
                image, organ_array, contrast, phase = self.preprocess_sample(idx)

                samples = [(image, organ_array)]

                if self.augment and row["split"] == "train":
                    samples = self.augmentation(image, organ_array, idx)

                for i, (img, mask) in enumerate(samples):
                    suffix = "" if i == 0 else f"_aug{i}"

                    save_path = row["output_path"].replace(".pt", f"{suffix}.pt")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    torch.save({
                        "patient_id": row["SubjectKeyRadiology"],
                        "key": row["MatchKey"],
                        "image": img,
                        "organ_mask": mask,
                        "contrast": contrast,
                        "phase": phase
                    }, save_path)

                    counter += 1

            except Exception as e:
                print(f"Skipping {idx}: {e}", flush=True)
                skipped += 1

        return counter, skipped

def save_final_dataset(dataset, output_root):
    output_root = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/contrast_preprocessed"

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

    dataset.to_csv("/projects/net_contrast_classification/contrast_phase/Preprocessing/Contrast_data/preprocessed_data.csv")

    return dataset
    


def main():
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"], 0)
    print(f"Running task {task_id}", flush=True)

    # -------- Load data --------
    data_dir = "/projects/net_contrast_classification/contrast_phase/data/cleaned_data_1.csv"
    dataset = files_load(data_dir, sample=None)

    train_idx, val_idx, test_idx = group_stratified_train_val_test_split(
                                                                        np.arange(len(dataset)),
                                                                        labels=dataset["phase"],
                                                                        groups=dataset["SubjectKeyRadiology"]
                                                                            )
    dataset = dataset.reset_index(drop=True)

    dataset["split"] = "unassigned"

    dataset.loc[train_idx, "split"] = "train"
    dataset.loc[val_idx, "split"] = "val"
    dataset.loc[test_idx, "split"] = "test"

    dataset["augment"] = ((dataset["split"] == "train") & (dataset["rare_class"] == 1)).astype(int)

    output_root = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/contrast_preprocessed"

    dataset = save_final_dataset(dataset, output_root)


    organ_ids = [1,2,3,5,8,9,13,51,52,63,64,65,66]

    preprocessor = Preprocess3D(
        dataset,
        organ_ids,
        pixdim=(1,1,1),
        resize=(128,128,128),
        augment=True
    )


    # -------- Assign split --------
    splits = ["train", "val", "test"]
    split_name = splits[task_id % 3]

    indices = dataset.index[dataset["split"] == split_name].to_numpy()

    # -------- Chunking --------
    chunks_per_split = 10
    chunk_id = task_id // 3
    chunks = np.array_split(indices, chunks_per_split)

    if chunk_id >= len(chunks):
        print("Nothing to process for this task.")
        return

    my_indices = chunks[chunk_id]

    print(f"Processing {split_name} | chunk {chunk_id} | size {len(my_indices)}", flush=True)

    preprocessor.process_and_save(my_indices)


if __name__ == "__main__":
    main()