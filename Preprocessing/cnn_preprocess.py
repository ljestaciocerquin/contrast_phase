from email.mime import image
import os, torch
import pandas as pd
import numpy as np
from pathlib import Path
from pathlib import PureWindowsPath
from monai.transforms import (Compose, LoadImage, Resized, Spacingd, CropForegroundd, ScaleIntensityRange, Resize, RandFlip, RandRotate90, RandGaussianNoise)
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from monai.data import MetaTensor

import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")

from Radiomics.radiomics_pipeline import multi_channel



def files_load(data_dir, sample = None):

    dataset = pd.read_csv(data_dir)

    dataset['server_folder'] = dataset.exist_on_server.apply(lambda x: "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET" 
                                                             if pd.notna(x)  
                                                             else "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/not_on_server")
    
    dataset = dataset[(dataset.contrast.isin(['Arterial', "Portal"]))
                & (dataset.is_liver_imaged == "Yes")
                & (dataset.phase_timing != '0.0')]

    dataset['contrast_timing'] = (dataset["contrast"].str.cat(dataset["phase_timing"], sep=" "))
    dataset["contrast_timing"] = dataset["contrast_timing"].astype(str).str.strip()

    if not sample:
        files = [os.path.join(row['server_folder'], PureWindowsPath(row['MatchKey']).name) for i, row in dataset.iterrows()]
        organ_files = [f.replace(".nii.gz", ".organs.nii.gz") for f in files]
        contrast = dataset["contrast"].tolist()
        phase = dataset["contrast_timing"].tolist()
        keys = dataset['MatchKey'].tolist()

    else:
        sampled_df = dataset.groupby('contrast_timing', group_keys=False).apply(lambda x: x.sample(n=min(len(x), sample), random_state=42))
        sampled_df['contrast_timing'] = sampled_df["contrast"].str.cat(sampled_df["phase_timing"], sep=" ")
        sampled_df["contrast_timing"] = sampled_df["contrast_timing"].astype(str).str.strip()

        files = [os.path.join(row['server_folder'], PureWindowsPath(row['MatchKey']).name) for i, row in sampled_df.iterrows()]
        organ_files = [f.replace(".nii.gz", ".organs.nii.gz") for f in files]
        contrast = sampled_df["contrast"].tolist()
        phase = sampled_df["contrast_timing"].tolist()
        keys = sampled_df['MatchKey'].tolist()

    return keys, files, organ_files, contrast, phase

class Preprocess3D:
    def __init__(self, keys, image_files, organ_files, contrast, phase, organ_ids,
                 test_size = 0.2, pixdim=(1,1,1), resize = (128,128,128), augment = False
                 ):
        
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
                                    RandFlip(prob=0.5, spatial_axis=0),
                                    RandFlip(prob=0.5, spatial_axis=1),
                                    RandFlip(prob=0.5, spatial_axis=2),
                                    RandRotate90(prob=0.5, spatial_axes=(0,1)),
                                    RandGaussianNoise(prob=0.2),
                                ])
        

    def data_split(self, stratify = "contrast"):
        if stratify == "contrast":
            labels = self.contrast
        elif stratify == "phase":
            labels = self.phase
        else:
            labels = None

        temp_idx, test_idx = train_test_split(
                                            np.arange(len(self.image_files)),
                                            test_size=self.test_size,             # 20% test
                                            random_state=42,                      # for reproducibility
                                            stratify=labels                  # maintain class balance
                                            )

        train_idx, val_idx = train_test_split(
                                            temp_idx,
                                            test_size=self.test_size,
                                            random_state=42,
                                            stratify=labels[temp_idx]
                                            )
        
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


    def preprocess_sample(self, idx, split="train"):

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
            print(f"Spacing done! Image shape:{image.shape}, Organ array shape: {organ_array.shape}")
        except Exception as e:
            print(f"Spacing failed: {e}")

        
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


        # ------------------ Augment -------------------
        if split == "train" and self.augment:
            image = self.augment_transform(image)    

            try:
                print(f"Augmentation done! Image shape:{image.shape}", flush=True)
            except Exception as e:
                print(f"Augmentation failed: {e}", flush=True)

        return image, organ_array, contrast, phase

    def process_and_save(self, indices, save_dir, split="train"):
        os.makedirs(save_dir, exist_ok=True)
        counter = 0
        skipped = 0
        keys = self.keys

        for _, idx in enumerate(indices):
            try:
                image, organ_array, contrast, phase = self.preprocess_sample(idx, split)

                save_path = os.path.join(save_dir, f"{keys[idx].replace(".nii.gz", "")}.pt")

                torch.save({
                    "key": keys[idx], 
                    "image": image,
                    "organ_mask": organ_array,
                    "contrast": contrast,
                    "phase": phase
                }, save_path)
                counter += 1
                print(f"Done processing sample {idx}", flush = True)


            except Exception as e:
                skipped += 1
                print(f"Skipping {idx}: {e}", flush=True)


        return counter, skipped

    def run(self, output_root, stratify = "contrast"):
        train_idx, val_idx, test_idx = self.data_split(stratify=stratify)
        print("Split done!\n", flush=True)

        train_count, train_skipped = self.process_and_save(train_idx, os.path.join(output_root, "train"), "train")
        print(f"\nTrain done! Saved {train_count}\n", flush=True)

        val_count, val_skipped = self.process_and_save(val_idx, os.path.join(output_root, "val"), "val")
        print(f"\nValidation done! Saved {val_count}\n", flush=True)

        test_count, test_skipped = self.process_and_save(test_idx, os.path.join(output_root, "test"), "test")
        print(f"\nTest done! Saved {test_count}\n", flush=True)

        total = train_count + val_count + test_count
        total_skipped = train_skipped + val_skipped + test_skipped
        print(f"\nTotal saved: {total}\nTotal skipped: {total_skipped}", flush=True)

class Preprocess2p5D:
    def __init__(self, image_files, organ_files, labels, organ_ids,
                test_size = 0.2, num_slices = 10, resize = (128,128,128), augment = False):
    
        self.image_files=image_files
        self.organ_files=organ_files
        self.labels=labels
        self.organ_ids=organ_ids
        self.test_size=test_size
        self.num_slices=num_slices
        self.resize=resize
        self.augment=augment

        self.loader=LoadImage(image_only=True, ensure_channel_first=True)
        self.base_transform = Compose([
                                    ScaleIntensityRanged(
                                                        a_min=-100, a_max=300,
                                                        b_min=0.0, b_max=1.0,
                                                        clip=True),
                                    Resize(self.resize)
                                    ])
        
        self.augment_transform = Compose([
                                    RandFlip(prob=0.5, spatial_axis=0),
                                    RandFlip(prob=0.5, spatial_axis=1),
                                    RandFlip(prob=0.5, spatial_axis=2),
                                    RandRotate90(prob=0.5, spatial_axes=(0,1)),
                                    RandGaussianNoise(prob=0.2),
                                ])
    
    def data_split(self):
        temp_idx, test_idx = train_test_split(
                                            np.arange(len(self.image_files)),
                                            test_size=self.test_size,        # 20% test
                                            random_state=42,                 # for reproducibility
                                            stratify=self.labels             # maintain class balance
                                            )

        train_idx, val_idx = train_test_split(
                                            temp_idx,
                                            test_size=self.test_size,
                                            random_state=42,
                                            stratify=self.labels[temp_idx]
                                            )
        
        return train_idx, val_idx, test_idx

    def get_2p5d_slices(self, image_array, organ_array):
        """
        Extract a 2.5D stack of slices around the organ center.
        
        Args:
            image_array: Tensor, shape (1, H, W, D)
            organ_array: Tensor or NumPy array, shape (C, H, W, D) or (H, W, D)
            num_slices: int, number of slices to extract
        
        Returns:
            slices: Tensor, shape (num_slices, H, W)
        """

        _, H, W, D = self.image_array.shape


        if organ_array is not None:
            organ_tensor = torch.as_tensor(organ_array, device=image_array.device)

            if organ_tensor.ndim > 3:
                organ_mask = (organ_tensor > 0).any(dim=0)  # (H, W, D)
            else:
                organ_mask = organ_tensor > 0

            # pick the slice with the most organ presence
            center = organ_mask.sum(dim=(0, 1)).argmax().item()

        else:
            center = image_array.sum(dim=(1, 2)).argmax().item()

        half = self.num_slices // 2

        if self.num_slices % 2 == 0:
            indices = [min(max(center + i, 0), D - 1) for i in range(-half, half)]
        else:
            indices = [min(max(center + i, 0), D - 1) for i in range(-half, half + 1)]

        indices = torch.tensor(indices, device=image_array.device)
        slices = image_array[0, :, :, indices]       # (H, W, K)
        slices = slices.permute(2, 0, 1)             # (K, H, W)

        return slices

    def preprocess_sample(self, idx, split="train"):

        image, organ_array, label = self.data_load(idx)

        image = self.get_2p5d_slices(organ_array)

        # -------- Intensity & Resizing --------

        image = self.base_transform(image)

        # -------- Augment --------
        if split == "train" and self.augment:
            image = self.augment_transform(image)       

        return image, label

def main():
    data_dir = "/projects/net_contrast_classification/contrast_phase/data/cleaned_data_1.csv"
    keys, image_files, organ_files, contrast, phase = files_load(data_dir, sample = 500)
    organ_ids = [1,2,3,5,8,9,13,51,52,63,64,65,66]

    preprocessor = Preprocess3D(
        keys,
        image_files,
        organ_files,
        contrast,
        phase,
        organ_ids,
        test_size=0.2,
        pixdim=(1,1,1),
        resize=(128,128,128),
        augment=False
    )

    print("Starting the preprocessing:", flush=True)
    output_dir = Path("/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/contrast_preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)  # creates if not exists

    preprocessor.run(str(output_dir), stratify="contrast")  # or stratify="phase" if you want to stratify by phase instead
    print("Preprocessing finished successfully!", flush=True)

if __name__ == "__main__":
    main()