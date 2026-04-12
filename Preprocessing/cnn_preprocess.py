
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
from sklearn.model_selection import GroupShuffleSplit



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

    dataset.loc[dataset["contrast"] == "Non-contrast", "phase_timing"] = None               # ensure NC cases have None phase_timing for safe concatenation

    dataset = contrast_timing(dataset)       # ensure NC cases have None contrast_timing for safe concatenation

    # ---------------- SAMPLE ----------------
    if sample is not None:
        dataset = (dataset.groupby('contrast_timing', dropna=False, group_keys=False)
                   .apply(lambda x: x.sample(n=min(len(x), sample), random_state=42))
                   .reset_index(drop=True))
        
        dataset = contrast_timing(dataset)   # re-apply contrast_timing after sampling to ensure consistency
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle rows
        


    # ---------------- FILE PATHS ----------------
    files = [os.path.join(row['server_folder'], PureWindowsPath(row['MatchKey']).name) for _, row in dataset.iterrows()]

    organ_files = [f.replace(".nii.gz", ".organs.nii.gz") for f in files]

    contrast = dataset["contrast"].tolist()
    phase = dataset["contrast_timing"].tolist()
    keys = dataset['MatchKey'].tolist()
    patient_id = dataset['SubjectKeyRadiology'].tolist()

    return patient_id, keys, files, organ_files, contrast, phase

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

class Preprocess3D:
    def __init__(self, patient_ids, keys, image_files, organ_files, contrast, phase, organ_ids,
                 test_size = 0.2, pixdim=(1,1,1), resize = (128,128,128), augment = False
                 ):
        
        self.patient_ids = patient_ids
        self.keys=np.array(keys)
        self.image_files=image_files
        self.organ_files=organ_files
        self.contrast=np.array(contrast)
        self.phase=np.array(phase)
        self.organ_ids=organ_ids
        self.test_size=test_size
        self.pixdim=pixdim
        self.resize=resize
        self.augment=augment
        self.rare_classes = {
                            # "Arterial Too Early",
                            "Portal Too Late",
                            "Portal Too Early",
                            "Arterial Too Late"
                             }

        self.loader=LoadImage(image_only=True, ensure_channel_first=True)
        self.spacing = Spacingd(
                                    keys=["image", "mask"],
                                    pixdim=(1,1,1),
                                    mode=["bilinear", "nearest"],
                                    allow_missing_keys=True)                # allow missing lesions key
        

        self.intensity_scaler = ScaleIntensityRange(a_min=-100, a_max=300, b_min=0.0, b_max=1.0, clip=True)
        
        self.crop = CropForegroundd(
                                    keys=["image", "mask"],     # what to crop
                                    source_key="mask",          # use mask to define bbox
                                    allow_missing_keys=True,
                                    margin = 20)
        

        self.resizer = Resized(
                            keys=["image", "mask"],
                            spatial_size=self.resize,
                            mode=["bilinear", "nearest"],
                            allow_missing_keys=True)                 # allow missing lesions key


        self.augment_transform = Compose([
                                        # RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0, allow_missing_keys=True), 
                                        # RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                        # RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2, allow_missing_keys=True),
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

    def data_split(self, image_files, patient_ids):
        all_indices = np.arange(len(image_files))

        # First split: train+val vs test
        temp_idx, test_idx = group_train_test_split(all_indices, test_size=0.2, groups=patient_ids)

        # Second split: train vs val
        train_idx, val_idx = group_train_test_split(temp_idx, test_size=0.2, groups=np.array(patient_ids)[temp_idx])

        return train_idx, val_idx, test_idx

    def data_load(self,idx):
        image = self.loader(self.image_files[idx])
        image_affine = image.affine
        organ_array = None
        contrast = self.contrast[idx]
        phase = self.phase[idx]


        if self.organ_ids is not None:
            try:
                organ_array_raw = self.loader(self.organ_files[idx])
                if organ_array_raw.ndim == 3:
                    organ_array_raw = organ_array_raw.unsqueeze(0)

                if np.size(organ_array_raw) == 0:
                    raise ValueError("Empty organ file")

                organ_array = multi_channel(self.organ_ids, organ_array_raw)
                # Convert to torch tensor
                organ_array = torch.from_numpy(organ_array.astype(np.float32))

                # Wrap as MetaTensor
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
    
    def augmentation(self, image, organ_array, phase, num_aug = 10):
        augmented_samples = []

        # always keep original
        augmented_samples.append((image, organ_array))

        if phase in self.rare_classes:
            for _ in range(num_aug):
                data = {"image": image.clone(), "mask": organ_array.clone()}
                data = self.augment_transform(data)
                augmented_samples.append((data["image"], data["mask"]))
        else:
            # common classes: occasionally augment
            if np.random.rand() < 0.2:                                      # 20% chance
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

    def process_and_save(self, indices, save_dir, split="train"):
        os.makedirs(save_dir, exist_ok=True)

        counter = 0
        skipped = 0

        for idx in indices:
            try:
                image, organ_array, contrast, phase = self.preprocess_sample(idx)

                base_name = self.keys[idx].replace(".nii.gz", "")

                if split == "train" and self.augment:
                    samples = self.augmentation(image, organ_array, phase, 10)
                    try:
                        print(f"Augmentation done! Generated {len(samples)} samples for phase: {phase}, {samples[0][0].shape}", flush=True)
                    except Exception as e:
                        print(f"Augmentation failed: {e}", flush=True)
                else:
                    samples = [(image, organ_array)]  # only original

                for i, (img, mask) in enumerate(samples):           # save augmented samples as separate files with suffix _aug1, _aug2, etc.
                    suffix = "" if i == 0 else f"_aug{i}"

                    save_path = os.path.join(save_dir, f"{base_name}{suffix}.pt")

                    torch.save({
                        "patient_id": self.patient_ids[idx],
                        "key": self.keys[idx],
                        "image": img,
                        "organ_mask": mask,
                        "contrast": contrast,
                        "phase": phase if not pd.isna(phase) else None
                    }, save_path)

                    counter += 1

            except Exception as e:
                print(f"Skipping {idx}: {e}", flush=True)
                skipped += 1

        return counter, skipped

    def run(self, output_root, split):  
        train_idx, val_idx, test_idx = self.data_split(self.image_files, self.patient_ids)
        print("Split done!\n", flush=True)

        if split == "train":
            count, skipped = self.process_and_save(
                train_idx, os.path.join(output_root, "train"), "train"
            )
            print(f"\nTrain done! Saved {count}, skipped {skipped}", flush=True)

        elif split == "val":
            count, skipped = self.process_and_save(
                val_idx, os.path.join(output_root, "val"), "val"
            )
            print(f"\nValidation done! Saved {count}, skipped {skipped}", flush=True)

        elif split == "test":
            count, skipped = self.process_and_save(
                test_idx, os.path.join(output_root, "test"), "test"
            )
            print(f"\nTest done! Saved {count}, skipped {skipped}", flush=True)


def main():
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    print(f"Running task {task_id}", flush=True)

    # -------- Load data --------
    data_dir = "/projects/net_contrast_classification/contrast_phase/data/cleaned_data_1.csv"
    patient_ids, keys, image_files, organ_files, contrast, phase = files_load(data_dir, sample=None)  

    organ_ids = [1,2,3,5,8,9,13,51,52,63,64,65,66]

    preprocessor = Preprocess3D(
        patient_ids,
        keys,
        image_files,
        organ_files,
        contrast,
        phase,
        organ_ids,
        test_size=0.2,
        pixdim=(1,1,1),
        resize=(128,128,128),
        augment=True             
    )

    # -------- Split once --------
    train_idx, val_idx, test_idx = preprocessor.data_split(image_files, patient_ids)

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
        print("Nothing to process for this task.")
        return

    my_indices = chunks[chunk_id]

    print(f"Processing {split_name} | chunk {chunk_id} | size {len(my_indices)}", flush=True)

    output_dir = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/contrast_preprocessed"

    preprocessor.process_and_save(
        my_indices,
        os.path.join(output_dir, split_name),
        split_name
    )

if __name__ == "__main__":
    main()