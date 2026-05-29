import os, torch
import pandas as pd
import numpy as np
from monai.transforms import (
    Compose, LoadImage, Resized, Spacingd, ScaleIntensityRanged
)
from monai.data import MetaTensor

import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")

from Radiomics.radiomics_pipeline import multi_channel
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class PairedDiffusionPreprocess:
    def __init__(self, dataset, organ_ids,
                 pixdim=(1, 1, 1), resize=(128, 128, 128)):

        self.dataset = dataset
        self.organ_ids = organ_ids
        self.pixdim = pixdim
        self.resize = resize

        self.loader = LoadImage(image_only=True, ensure_channel_first=True)

        self.image_keys = ["a_image", "p_image"]
        self.mask_keys = ["a_liver", "p_liver"]

        self.transforms = Compose([
            Spacingd(keys=self.image_keys, pixdim=self.pixdim, mode="bilinear", allow_missing_keys=True),
            Spacingd(keys=self.mask_keys, pixdim=self.pixdim, mode="nearest", allow_missing_keys=True),

            Resized(keys=self.image_keys, spatial_size=self.resize, mode="bilinear", allow_missing_keys=True),
            Resized(keys=self.mask_keys, spatial_size=self.resize, mode="nearest", allow_missing_keys=True),

            ScaleIntensityRanged(
                keys=self.image_keys,
                a_min=-100, a_max=300,
                b_min=0.0, b_max=1.0,
                clip=True,
                allow_missing_keys=True
            )
        ])

    def safe_load(self, path, required=True):
        if path is None or pd.isna(path):
            if required:
                raise FileNotFoundError(f"[INVALID PATH] {path}")
            return None

        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"[MISSING FILE] {path}")
            return None

        try:
            return self.loader(path)
        except Exception as e:
            if required:
                raise RuntimeError(f"[LOAD FAILED] {path} | {e}")
            return None

    def process_organs(self, mask_raw, affine, already_single_channel=False):
        if mask_raw is None:
            return None

        try:
            if mask_raw.ndim == 3:
                mask_raw = mask_raw.unsqueeze(0)

            if np.size(mask_raw) == 0:
                return None

            if already_single_channel:
                organ_array = mask_raw
            else:
                organ_array = multi_channel(self.organ_ids, mask_raw)

            organ_array = torch.from_numpy(organ_array.astype(np.float32))
            return MetaTensor(organ_array, affine=affine)

        except Exception as e:
            print(f"Organ processing failed: {e}")
            return None

    def data_load(self, idx):
        row = self.dataset.iloc[idx]
        split = row["split"]
        is_inference = split == "inference"

        # -------- images --------
        a_image = self.safe_load(row["arterial_image"], required=not is_inference)
        p_image = self.safe_load(row["portal_image"], required=not is_inference)

        # STRICT: train/val/test must be complete pairs
        if not is_inference:
            if a_image is None or p_image is None:
                raise ValueError(f"Incomplete pair in {split} at idx {idx}")

        # inference: allow partial, but not fully empty
        if is_inference:
            if a_image is None and p_image is None:
                raise ValueError(f"Both modalities missing at idx {idx}")

        a_affine = a_image.affine if a_image is not None else None
        p_affine = p_image.affine if p_image is not None else None

        # -------- masks (always optional) --------
        a_organs_raw = self.safe_load(row["arterial_organs"], required=False)
        p_organs_raw = self.safe_load(row["portal_organs"], required=False)

        a_liver = self.process_organs(a_organs_raw, a_affine)
        p_liver = self.process_organs(p_organs_raw, p_affine, already_single_channel=True)

        return a_image, p_image, a_liver, p_liver

    def cropping(self, image, organ_array):
        if image is None:
            return None, None, None

        if organ_array is None:
            return image, organ_array, None

        foreground_mask = torch.from_numpy(
            (organ_array.sum(axis=0) > 0).astype(np.float32)
        )

        mask_nonzero = torch.nonzero(foreground_mask)

        if mask_nonzero.numel() == 0:
            zmin, ymin, xmin = 0, 0, 0
            zmax, ymax, xmax = foreground_mask.shape
        else:
            zmin, ymin, xmin = mask_nonzero.min(0)[0]
            zmax, ymax, xmax = mask_nonzero.max(0)[0] + 1

        margin = 20
        zmin = max(zmin - margin, 0)
        ymin = max(ymin - margin, 0)
        xmin = max(xmin - margin, 0)

        zmax = min(zmax + margin, foreground_mask.shape[0])
        ymax = min(ymax + margin, foreground_mask.shape[1])
        xmax = min(xmax + margin, foreground_mask.shape[2])

        image = image[:, zmin:zmax, ymin:ymax, xmin:xmax]
        organ_array = organ_array[:, zmin:zmax, ymin:ymax, xmin:xmax]

        return image, organ_array, None

    def preprocess_sample(self, idx):

        a_image, p_image, a_liver, p_liver = self.data_load(idx)

        # -------- cropping (independent per modality) --------
        if a_liver is not None:
            a_image, a_liver, _ = self.cropping(a_image, a_liver)

        if p_liver is not None:
            p_image, p_liver, _ = self.cropping(p_image, p_liver)

        data = {}

        if a_image is not None:
            data["a_image"] = a_image
        if p_image is not None:
            data["p_image"] = p_image

        if a_liver is not None:
            data["a_liver"] = a_liver
        if p_liver is not None:
            data["p_liver"] = p_liver

        print("Cropping done!", flush =True)

        data = self.transforms(data)

        return (
            data.get("a_image"),
            data.get("p_image"),
            data.get("a_liver"),
            data.get("p_liver")
        )

    def process_and_save(self, indices):
        counter = 0
        skipped = 0

        for idx in indices:
            row = self.dataset.iloc[idx]

            try:
                a_img, p_img, a_liver, p_liver = self.preprocess_sample(idx)

                save_path = row["output_path"]
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                torch.save({
                    "patient_id": row["SubjectKeyRadiology"],
                    "exam_date": row["ExamDate"],
                    "arterial_image": a_img,
                    "portal_image": p_img,
                    "arterial_liver": a_liver,
                    "portal_liver": p_liver
                }, save_path)

                counter += 1

            except Exception as e:
                print(f"Skipping {idx}: {e}", flush=True)
                skipped += 1

        return counter, skipped


def main():
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"], 0)

    data_dir = "/projects/net_contrast_classification/contrast_phase/Preprocessing/Paired_diffusion_data/full_pairs.csv"
    dataset = pd.read_csv(data_dir)

    organ_ids = [5]

    preprocessor = PairedDiffusionPreprocess(
        dataset,
        organ_ids,
        pixdim=(1, 1, 1),
        resize=(128, 128, 128)
    )

    # splits = ["train", "val", "test", "inference"]
    # split_name = splits[task_id % 4]
    split_name = "inference"  # change if needed

    indices = dataset.index[dataset["split"] == split_name].to_numpy()

    chunks = np.array_split(indices, 2)
    chunk_id = task_id // 4

    if chunk_id >= len(chunks):
        print("Nothing to process")
        return

    my_indices = chunks[chunk_id]

    print(f"Processing {split_name} | chunk {chunk_id} | size {len(my_indices)}")

    counter, skipped = preprocessor.process_and_save(my_indices)

    print(f"Processed {counter} | Skipped {skipped}")


if __name__ == "__main__":
    main()