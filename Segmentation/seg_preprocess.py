import pandas as pd
import numpy as np
from pathlib import Path
import os, random, torch
from pathlib import PureWindowsPath
from monai.transforms import LoadImage
from monai.data import Dataset, DataLoader
from monai.transforms import (Compose, LoadImage, Spacing, Resized, Spacingd, CropForegroundd, ResizeWithPadOrCrop, ResizeWithPadOrCropd, ScaleIntensityRange, Resize, RandFlip, RandRotate90, RandGaussianNoise)
from monai.data import MetaTensor
import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")
from Preprocessing.cnn_preprocess import Preprocess3D
from Radiomics.radiomics_pipeline import multi_channel
from sklearn.model_selection import train_test_split



def files_load(data_dir, sample = None):

    dataset = pd.read_csv(data_dir)

    dataset['server_folder'] = dataset.exist_on_server.apply(lambda x: "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET" 
                                                             if pd.notna(x)  
                                                             else "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/not_on_server")
    
    dataset = dataset[(dataset.contrast.isin(['Arterial', "Portal"]))
                & (dataset.is_liver_imaged == "Yes")
                & (dataset.phase_timing != '0.0')
                & (dataset.is_lesionfree == "No") # remove later
                ]
    
    dataset = dataset.sort_values(by=["SubjectKeyRadiology", "ExamDate", "contrast"], ascending=True)

    if not sample:
        files = [os.path.join(row['server_folder'], PureWindowsPath(row['MatchKey']).name) for _, row in dataset.iterrows()]
        organ_files = [f.replace(".nii.gz", ".organs.nii.gz") for f in files]
        seg_files = [f.replace(".nii.gz", ".seg.nii.gz") for f in files]
        labels = dataset['contrast'].tolist()
        keys = dataset['MatchKey'].tolist()

    else:
        sampled_df = (
            dataset
            .groupby('contrast', group_keys=False)
            .sample(n=sample, replace=True, random_state=42)
            .reset_index(drop=True)
        )

        files = [os.path.join(row['server_folder'], PureWindowsPath(row['MatchKey']).name) for _, row in sampled_df.iterrows()]
        organ_files = [f.replace(".nii.gz", ".organs.nii.gz") for f in files]
        seg_files = [f.replace(".nii.gz", ".seg.nii.gz") for f in files]
        labels = sampled_df['contrast'].tolist()
        keys = sampled_df['MatchKey'].tolist()


    return keys, files, organ_files, seg_files, labels

class PreprocessSeg():
    def __init__(self, keys, image_files, organ_files, seg_files, labels, organ_ids,
                test_size = 0.2, pixdim=(1,1,1), resize = (128,128,128)
                ):
        
        self.keys=np.array(keys)
        self.image_files=image_files
        self.organ_files=organ_files
        self.seg_files = seg_files
        self.labels=np.array(labels)
        self.organ_ids=organ_ids
        self.test_size=test_size
        self.pixdim=pixdim
        self.resize=resize


        self.loader = LoadImage(image_only=True, ensure_channel_first=True)

        self.spacing = Spacingd(
                                    keys=["image", "mask", "lesions"],
                                    pixdim=(1,1,1),
                                    mode=["bilinear", "nearest", "nearest"],
                                    allow_missing_keys=True)                # allow missing lesions key
        
        self.intensity_scaler = ScaleIntensityRange(a_min=-100, a_max=300, b_min=0.0, b_max=1.0, clip=True)

        self.resizer = Resized(
                                    keys=["image", "mask", "lesions"],
                                    spatial_size=self.resize,
                                    mode=["bilinear", "nearest", "nearest"],
                                    allow_missing_keys=True)                 # allow missing lesions key
        
        # self.cropper = CropForegroundd(                                    # FIX: REMOVE DICT, CROP ONLY THE IMAGE?
        #                             keys=["image", "mask", "lesions"],     # what to crop
        #                             source_key="mask",                     # use mask to define bbox
        #                             select_fn=lambda x: x > 0,             # ensures detection
        #                             margin = 20)
        


        # self.register = # ADD IMAGE REGISTRATION

    def data_split(self):


        temp_idx, test_idx = train_test_split(
                                            np.arange(len(self.image_files)),
                                            test_size=self.test_size,             # 20% test
                                            random_state=42,                      # for reproducibility
                                            stratify=self.labels                  # maintain class balance
                                            )

        train_idx, val_idx = train_test_split(
                                            temp_idx,
                                            test_size=self.test_size,
                                            random_state=42,
                                            stratify=self.labels[temp_idx]
                                            )
        
        return train_idx, val_idx, test_idx

    def data_load(self,idx):
        image = self.loader(self.image_files[idx])
        image_affine = image.affine
        lesions_raw = self.loader(self.seg_files[idx])
        organ_array, lesions = None, None
        label = self.labels[idx]

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

        if lesions_raw is not None and np.size(lesions_raw) > 0:
            lesions = torch.from_numpy((lesions_raw > 0).astype(np.float32))
            if lesions.ndim == 3:  # add channel dim
                lesions = lesions.unsqueeze(0)
            lesions = MetaTensor(lesions, affine=image_affine)
        else:
            lesions = None

        
        return image, organ_array, lesions, label
    
    def cropping(self, image, organ_array, lesions):
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

        # Crop image, organ_array, lesions
        image = image[:, zmin:zmax, ymin:ymax, xmin:xmax]
        organ_array = organ_array[:, zmin:zmax, ymin:ymax, xmin:xmax]
        if lesions is not None:
            lesions = lesions[:, zmin:zmax, ymin:ymax, xmin:xmax]

        return image, organ_array, lesions

    def preprocess_sample(self, idx):
        # -------- Load data --------
        image, organ_array, lesions, label = self.data_load(idx)

        # -------- Crop using organ mask --------

        if organ_array is not None:
            image, organ_array, lesions = self.cropping(image, organ_array, lesions)
            try:
                print(f"Cropping done! Image: {image.shape}, Organ array: {organ_array.shape}, Lesions: {lesions.shape if lesions is not None else 'N/A'}", flush=True)
            except Exception as e:
                print(f"Cropping failed: {e}", flush=True)

        # -------- Resampling --------

        data = {"image": image, "mask": organ_array, "lesions": lesions}
        data = self.spacing(data)
        image = data["image"]
        organ_array = data["mask"]
        lesions = data.get("lesions", None)

        try:
            print(f"Spacing done! Image shape:{image.shape}, Organ array shape: {organ_array.shape}, Lesions shape: {lesions.shape if lesions is not None else 'N/A'}", flush=True)
        except Exception as e:
            print(f"Spacing failed: {e}", flush=True)

        # -------- Intensity scaling --------
    
        image = self.intensity_scaler(image)
        try:
            print(f"Intensity scaling done! Image shape:{image.shape}", flush=True)
        except Exception as e:
            print(f"Intensity scaling failed: {e}", flush=True)

        # -------- Resizing --------

        data = {"image": image, "mask": organ_array, "lesions": lesions}
        data = self.resizer(data)
        image = data["image"]
        organ_array = data["mask"]
        lesions = data.get("lesions", None)

        try:
            print(f"Resizing successful! Image: {image.shape}, Organ array: {organ_array.shape}, Lesions: {lesions.shape if lesions is not None else 'N/A'}", flush=True)
        except Exception as e:
            print(f"Resizing failed: {e}", flush=True)


        # -------- Debug print --------
        print(f"Image shape: {image.shape}")
        if organ_array is not None:
            print(f"Organ array shape: {organ_array.shape}")
        if lesions is not None:
            print(f"Lesions shape: {lesions.shape}")

        return image, organ_array, lesions, label

    def process_and_save(self, indices, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        counter = 0
        skipped = 0
        keys = self.keys

        for _, idx in enumerate(indices):
            try:
                image, organ_array, lesions, label = self.preprocess_sample(idx)

                save_path = os.path.join(save_dir, f"{keys[idx].replace(".nii.gz", "_seg")}.pt")

                torch.save({
                    "key": keys[idx], 
                    "image": image,
                    "organ_array": organ_array,
                    "lesions":lesions,
                    "label": label
                }, save_path)
                counter += 1
                print(f"Done processing sample {idx}", flush = True)


            except Exception as e:
                skipped += 1
                print(f"Skipping {idx}: {e}", flush=True)


        return counter, skipped
    
    def run(self, output_root):
        train_idx, val_idx, test_idx = self.data_split()
        print("Split done!\n", flush=True)


        train_count, train_skipped = self.process_and_save(train_idx, os.path.join(output_root, "train"))
        print(f"\nTrain done! Saved {train_count}\n", flush=True)

        val_count, val_skipped = self.process_and_save(val_idx, os.path.join(output_root, "val"))
        print(f"\nValidation done! Saved {val_count}\n", flush=True)

        test_count, test_skipped = self.process_and_save(test_idx, os.path.join(output_root, "test"))
        print(f"\nTest done! Saved {test_count}\n", flush=True)

        total = train_count + val_count + test_count
        total_skipped = train_skipped + val_skipped + test_skipped
        print(f"\nTotal saved: {total}\nTotal skipped: {total_skipped}", flush=True)

def main():
    data_dir = "/projects/net_contrast_classification/contrast_phase/data/cleaned_data_1.csv"
    keys, image_files, organ_files, seg_files, labels = files_load(data_dir, sample = 300)
    organ_ids = [5]   # liver ID in organ segmentation maps

    preprocessor = PreprocessSeg(
        keys,
        image_files,
        organ_files,
        seg_files,
        labels,
        organ_ids,
        test_size=0.2,
        pixdim=(1,1,1),
        resize=(128,128,128)
    )

    print("Starting the preprocessing:", flush=True)
    output_dir = Path("/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/seg_preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)  # creates if not exists

    preprocessor.run(str(output_dir))

    print("Preprocessing finished successfully!", flush=True)

if __name__ == "__main__":
    main()

