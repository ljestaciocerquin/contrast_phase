import os, torch
import pandas as pd
import numpy as np
from monai.transforms import (Compose, LoadImage, Resized, Spacingd, ScaleIntensityRanged)
import torch.nn.functional as F
from monai.data import MetaTensor

import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")

from contrast_phase.Radiomics.radiomics_extract import multi_channel
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class PairedPreprocess:
    def __init__(self, dataset, organ_ids,
            pixdim=(1,1,1), resize = (128,128,128)
            ):
    
        self.dataset = dataset
        self.organ_ids=organ_ids
        self.pixdim=pixdim
        self.resize=resize
        self.loader=LoadImage(image_only=True, ensure_channel_first=True)
        self.image_keys = ["a_image", "p_image"]
        self.mask_keys = ["a_mask", "p_mask"]
        self.transforms = Compose([
                                Spacingd(keys=self.image_keys, pixdim=self.pixdim, mode="bilinear"),
                                Spacingd(keys=self.mask_keys, pixdim=self.pixdim, mode="nearest", allow_missing_keys=True),

                                Resized(keys=self.image_keys, spatial_size=self.resize, mode="bilinear"),
                                Resized(keys=self.mask_keys, spatial_size=self.resize, mode="nearest", allow_missing_keys=True),

                                ScaleIntensityRanged(
                                    keys=self.image_keys,
                                    a_min=-100, a_max=300,
                                    b_min=0.0, b_max=1.0,
                                    clip=True,
                                )
                            ])

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
        row = self.dataset.iloc[idx]

        # -------- Load images --------
        a_image = self.safe_load(row["arterial_file"])
        p_image = self.safe_load(row["portal_file"])

        a_affine = a_image.affine if a_image is not None else None
        p_affine = p_image.affine if p_image is not None else None

        # -------- Load raw masks --------
        a_organs_raw = self.safe_load(row["arterial_organs"])
        p_organs_raw = self.safe_load(row["portal_organs"])

        # -------- Process masks --------
        a_organs = self.process_organs(a_organs_raw, a_affine)
        p_organs = self.process_organs(p_organs_raw, p_affine)

        times = row["time_interval"]

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


        # -------- Cropping --------
        if a_organs is not None and p_organs is not None:
            try:
                a_image, a_organs, _ = self.cropping(a_image, a_organs)
                p_image, p_organs, _ = self.cropping(p_image, p_organs)

            except Exception as e:
                print(f"Cropping failed: {e}", flush=True)


        # -------- MONAI input --------
        data = {
            "a_image": a_image,
            "p_image": p_image,
        }

        if a_organs is not None:
            data["a_mask"] = a_organs

        if p_organs is not None:
            data["p_mask"] = p_organs

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
    

    def process_and_save(self, indices):
        counter = 0
        skipped = 0

        for idx in indices:
            row = self.dataset.iloc[idx]

            try:
                a_img, p_img, a_mask, p_mask, interval = self.preprocess_sample(idx)

                samples = [(a_img, p_img, a_mask, p_mask, interval)]

                for (a_i, p_i, a_m, p_m, t) in samples:

                    save_path = row["output_path"]
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    # -------- save --------
                    torch.save({
                        "patient_id": row["SubjectKeyRadiology"],
                        "exam_date": row["ExamDate"],

                        "arterial_image": a_i,
                        "portal_image": p_i,

                        "arterial_mask": a_m,
                        "portal_mask": p_m,

                        "time_interval": t
                    }, save_path)

                    counter += 1

            except Exception as e:
                print(f"Skipping {idx}: {e}", flush=True)
                skipped += 1

        return counter, skipped

def main():
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"], 0)
    print(f"Running task {task_id}", flush=True)
    
    # -------- Load paired data --------
    data_dir = "/projects/net_contrast_classification/contrast_phase/Preprocessing/Time_interval_data/paired_preprocessed_data.csv"
    dataset = pd.read_csv(data_dir)


    organ_ids = [1,2,3,5,8,9,13,51,52,63,64,65,66]

    preprocessor = PairedPreprocess(
        dataset,
        organ_ids,
        pixdim=(1,1,1),
        resize=(128,128,128)
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

    # -------- Output --------
    preprocessor.process_and_save(my_indices)


if __name__ == "__main__":
    main()