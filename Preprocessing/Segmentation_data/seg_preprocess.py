import os, torch
import pandas as pd
import numpy as np
from monai.transforms import (
    Compose, LoadImage, Spacingd, ScaleIntensityRanged, ResizeWithPadOrCropd
)
from monai.data import MetaTensor
import torch.nn.functional as F

import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")

from contrast_phase.Radiomics.radiomics_extract import multi_channel
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class ORMaskCropper:

    def __init__(self, margin=20):

        self.margin = margin

    # ----------------------------
    # shape alignment (CRITICAL)
    # ----------------------------

    def _match_shape(self, tensor, reference):

        if tensor is None or reference is None:
            return tensor

        if tensor.shape[1:] == reference.shape[1:]:
            return tensor

        return F.interpolate(
            tensor.unsqueeze(0).float(),
            size=reference.shape[1:],
            mode="nearest",
        ).squeeze(0)

    # ----------------------------
    # build OR mask (liver OR lesion)
    # ----------------------------

    def build_or_mask(self, masks, reference):
        foreground = None

        for m in masks:
            if m is None:
                continue

            m = self._match_shape(m, reference)
            if m.ndim == 3:
                m = m.unsqueeze(0)

            m = (m > 0)[0]  # [Z,Y,X]

            foreground = m if foreground is None else (foreground | m)  # OR mask

        return foreground

    # ----------------------------
    # bbox with SAFE margin handling
    # ----------------------------

    def compute_bbox(self, foreground):

        if foreground is None:
            return None

        coords = torch.nonzero(foreground)

        if coords.numel() == 0:
            return None

        zmin, ymin, xmin = coords.min(0)[0]
        zmax, ymax, xmax = coords.max(0)[0] + 1     

        zmin = int(max(zmin - self.margin, 0))      # apply margin = 20 pixels
        ymin = int(max(ymin - self.margin, 0))
        xmin = int(max(xmin - self.margin, 0))

        zmax = int(zmax + self.margin)
        ymax = int(ymax + self.margin)
        xmax = int(xmax + self.margin)

        return zmin, zmax, ymin, ymax, xmin, xmax

    # ----------------------------
    # safe crop (NO silent truncation bugs)
    # ----------------------------

    def crop(self, tensor, bbox):

        if tensor is None:
            return None

        zmin, zmax, ymin, ymax, xmin, xmax = bbox

        C, Z, Y, X = tensor.shape

        # clamp properly (IMPORTANT FIX)

        zmin = max(0, zmin); zmax = min(Z, zmax)
        ymin = max(0, ymin); ymax = min(Y, ymax)
        xmin = max(0, xmin); xmax = min(X, xmax)

        if zmax <= zmin or ymax <= ymin or xmax <= xmin:

            return None

        affine = tensor.affine if isinstance(tensor, MetaTensor) else None
        cropped = tensor[:, zmin:zmax, ymin:ymax, xmin:xmax]

        if affine is not None:
            cropped = MetaTensor(cropped.clone(), affine=affine)

        return cropped

    # ----------------------------
    # HARD CHECK (prevents silent lesion loss)
    # ----------------------------

    def verify_containment(self, lesion, bbox, name="lesion"):

        if lesion is None:
            return True

        zmin, zmax, ymin, ymax, xmin, xmax = bbox

        coords = torch.nonzero((lesion > 0).detach().cpu())

        if coords.numel() == 0:
            return True

        z, y, x = coords[:,1], coords[:,2], coords[:,3]

        ok = (
            (z.min() >= zmin) and (z.max() < zmax) and
            (y.min() >= ymin) and (y.max() < ymax) and
            (x.min() >= xmin) and (x.max() < xmax)
        )

        if not ok:
            raise ValueError(f"[WARNING] {name} NOT fully inside crop!")

        return ok

    # ----------------------------
    # FULL PIPELINE
    # ----------------------------

    def crop_all(self,
                 a_image, p_image,
                 a_liver, p_liver,
                 a_lesion, p_lesion):

        reference = a_image if a_image is not None else p_image

        if reference is None:

            return None, None, None, None, None, None

        masks = [a_liver, p_liver, a_lesion, p_lesion]

        foreground = self.build_or_mask(masks, reference)

        if foreground is None:
            return None, None, None, None, None, None

        bbox = self.compute_bbox(foreground)

        # sanity check - ensure the lesion is inside the cropping box

        self.verify_containment(a_lesion, bbox, "arterial lesion")
        self.verify_containment(p_lesion, bbox, "portal lesion")

        if bbox is None:
            return None, None, None, None, None, None

        # apply same crop everywhere

        a_image  = self.crop(a_image, bbox)
        p_image  = self.crop(p_image, bbox)

        a_liver  = self.crop(a_liver, bbox)
        p_liver  = self.crop(p_liver, bbox)

        a_lesion = self.crop(a_lesion, bbox)
        p_lesion = self.crop(p_lesion, bbox)


        return a_image, p_image, a_liver, p_liver, a_lesion, p_lesion

class SegPreprocess:
    def __init__(self, dataset, organ_ids,
                 pixdim=(1, 1, 1), 
                #  resize=(128, 128, 128)
                 ):

        self.dataset = dataset
        self.organ_ids = organ_ids
        self.pixdim = pixdim
        # self.resize = resize

        self.loader = LoadImage(image_only=True, ensure_channel_first=True)

        self.image_keys = ["a_image", "p_image"]
        self.liver_keys = ["a_liver", "p_liver"]
        self.lesion_keys = ["a_lesion", "p_lesion"]


        self.transforms = Compose([
            Spacingd(keys=self.image_keys, pixdim=self.pixdim, mode=3, allow_missing_keys=True),        # cubic spline
            Spacingd(keys=self.liver_keys, pixdim=self.pixdim, mode="nearest", allow_missing_keys=True, padding_mode = "zeros", align_corners=True),
            Spacingd(keys=self.lesion_keys, pixdim=self.pixdim, mode="nearest", allow_missing_keys=True, padding_mode = "zeros", align_corners=True),

            # ResizeWithPadOrCropd(keys = self.image_keys, spatial_size = self.resize, allow_missing_keys = True),
            # ResizeWithPadOrCropd(keys = self.liver_keys, spatial_size = self.resize, allow_missing_keys = True),
            # ResizeWithPadOrCropd(keys = self.lesion_keys, spatial_size = self.resize, allow_missing_keys = True),


            ScaleIntensityRanged(
                keys=self.image_keys,
                a_min=-70, a_max=180,
                b_min=0.0, b_max=1.0,
                clip=True,
                allow_missing_keys=True
            )
        ])

    def fix_wrong_shape(self, a_mask, p_mask, wrong_shape = (1,1,1)):
        if (
            a_mask is not None 
            and p_mask is not None 
            and a_mask.max().item() == 0.0
            and tuple(a_mask.shape[-3:]) == wrong_shape):

            print(f"AP lesion has shape {wrong_shape}", flush = True)
            a_mask = MetaTensor(torch.zeros_like(p_mask), affine=p_mask.affine)

        elif (
            a_mask is not None
            and p_mask is not None
            and p_mask.max().item() == 0.0
            and tuple(p_mask.shape[-3:]) == wrong_shape):

            print(f"PVP lesion has shape {wrong_shape}", flush = True)
            p_mask = MetaTensor(torch.zeros_like(a_mask), affine=a_mask.affine)

        return a_mask, p_mask



    def lesion_volume(self, lesion):
        """
        Returns lesion physical volume in mm3.
        """
        if lesion is None:
            return 0.0

        spacing = lesion.pixdim
        print(f"Spacing: {spacing}")
        voxels = (lesion > 0).float().sum().item()
        volume = voxels * spacing[0] * spacing[1] * spacing[2]

        return volume.item()
    
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

            if torch.numel(mask_raw) == 0:
                print("Empty liver mask", flush=True)
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

    def process_lesions(self, mask_raw, modality = "arterial"):
        if mask_raw is None:
            print("Lesion mask is None", flush = True)
            return None

        try:
            if mask_raw.ndim == 3:
                mask_raw = mask_raw.unsqueeze(0)

                
            if mask_raw.max() == 0:
                print(f"Empty {modality} lesion mask", flush=True)

                return MetaTensor(
                    torch.zeros_like(mask_raw, dtype=torch.float32),
                    affine=mask_raw.affine if isinstance(mask_raw, MetaTensor) else None
                )

            return MetaTensor(
                    mask_raw.float(),
                    affine=mask_raw.affine if isinstance(mask_raw, MetaTensor) else None
                )

        except Exception as e:
            print(f"Lesion processing failed: {e}")
            return None

    def data_load(self, idx):
        row = self.dataset.iloc[idx]
        split = row["split"]
        is_inference = split == "inference"

        a_image = self.safe_load(row["arterial_image"], required=not is_inference)
        p_image = self.safe_load(row["portal_image"], required=not is_inference)

        # enforce inference constraint
        if is_inference and a_image is None and p_image is None:
            raise ValueError(f"Both modalities missing at idx {idx}")

        # ALWAYS define outputs
        a_liver = p_liver = None
        a_lesion = p_lesion = None

        def build_modality(image, aff_key="arterial"):
            if image is None:
                return None, None, None

            affine = image.affine
            organs_raw = self.safe_load(row[f"{aff_key}_organs"], required= not is_inference)
            liver = self.process_organs(organs_raw, affine) if aff_key == "arterial" else self.process_organs(organs_raw, affine, already_single_channel=True)

            lesion_raw = self.safe_load(row[f"{aff_key}_lesion"], required= not is_inference)
            lesion = self.process_lesions(lesion_raw, modality=aff_key)

            return image, liver, lesion

        a_image, a_liver, a_lesion = build_modality(a_image, "arterial")
        p_image, p_liver, p_lesion = build_modality(p_image, "portal")


        # FIX AP LESIONS THAT HAVE SHAPE (1,1,1) => SAME SHAPE AS THE PVP LESION

        a_lesion, p_lesion = self.fix_wrong_shape(a_lesion, p_lesion)

        return a_image, p_image, a_liver, p_liver, a_lesion, p_lesion

    def preprocess_sample(self, idx):
        print("\n_____________________________")

        a_image, p_image, a_liver, p_liver, a_lesion, p_lesion = self.data_load(idx)

        print(
            f"\nLoading done! "
            f"a_image: {a_image.shape if a_image is not None else None}, "
            f"p_image: {p_image.shape if p_image is not None else None}, "
            f"a_liver: {a_liver.shape if a_liver is not None else None}, "
            f"p_liver: {p_liver.shape if p_liver is not None else None}, "
            f"a_lesion: {a_lesion.shape if a_lesion is not None else None}, "
            f"p_lesion: {p_lesion.shape if p_lesion is not None else None}",
            flush=True,
        )

        a_vol_before = self.lesion_volume(a_lesion)
        p_vol_before = self.lesion_volume(p_lesion)

        print(f"Initial lesion volume (mm3): arterial = {round(a_vol_before, 2)}, portal = {round(p_vol_before, 2)}", flush=True)


        # Cropping
        cropper = ORMaskCropper(margin=20)
        try:
            a_image, p_image, a_liver, p_liver, a_lesion, p_lesion = cropper.crop_all(a_image, p_image, a_liver, p_liver, a_lesion, p_lesion)
            print(f"\nCropping done!", flush=True)
        
        except ValueError as e:
            print(f"\nCropping failed: {e}")
        


        a_vol_after = self.lesion_volume(a_lesion)
        p_vol_after = self.lesion_volume(p_lesion)

        # Check that lesions have not disappeared

        if a_vol_before > 0 and a_vol_after == 0:
            raise ValueError(
                f"Arterial lesion disappeared during preprocessing "
                f"(before={a_vol_before}, after={a_vol_after}, idx={idx})"
            )

        if p_vol_before > 0 and p_vol_after == 0:
            raise ValueError(
                f"Portal lesion disappeared during preprocessing "
                f"(before={p_vol_before}, after={p_vol_after}, idx={idx})"
            )
        
        print(f"Lesion volume change after cropping (delta mm3): arterial = {round(abs(a_vol_before-a_vol_after), 2)}, portal = {round(abs(p_vol_before-p_vol_after), 2)}", flush=True)


        data = {}

        if a_image is not None:
            data["a_image"] = a_image
        if p_image is not None:
            data["p_image"] = p_image
        if a_liver is not None:
            data["a_liver"] = a_liver
        if p_liver is not None:
            data["p_liver"] = p_liver
        if a_lesion is not None:
            data["a_lesion"] = a_lesion
        if p_lesion is not None:
            data["p_lesion"] = p_lesion

        if len(data) == 0:
            raise ValueError("No valid modalities for sample")

        try:
            data = self.transforms(data)
            print(f"\nTransforms done! a_image: {data['a_image'].shape if 'a_image' in data else 'None'}, ", 
                  f"p_image: {data['p_image'].shape if 'p_image' in data else 'None'}, ",
                  f"a_liver: {data['a_liver'].shape if 'a_liver' in data else 'None'}, ",
                  f"p_liver: {data['p_liver'].shape if 'p_liver' in data else 'None'}, ",
                  f"a_lesion: {data['a_lesion'].shape if 'a_lesion' in data else 'None'}, ",
                  f"p_lesion: {data['p_lesion'].shape if 'p_lesion' in data else 'None'}", flush=True)

        except Exception as e:
            print(f"Transforms failed keys={list(data.keys())}: {e}")
            raise
            # return (None,) * 6

        a_vol_after = self.lesion_volume(data.get("a_lesion"))
        p_vol_after = self.lesion_volume(data.get("p_lesion"))

        print(f"Lesion volume change after transforms (delta mm3): arterial = {round(abs(a_vol_before-a_vol_after), 2)}, portal = {round(abs(p_vol_before-p_vol_after), 2)}", flush=True)

        return (
            data.get("a_image"),
            data.get("p_image"),
            data.get("a_liver"),
            data.get("p_liver"),
            data.get("a_lesion"),
            data.get("p_lesion"),
        )

    def process_and_save(self, indices):
        counter = 0
        skipped = 0

        for idx in indices:
            row = self.dataset.iloc[idx]
            try:
                a_img, p_img, a_liver, p_liver, a_lesion, p_lesion = self.preprocess_sample(idx)

                save_path = row["output_path"]
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                torch.save({
                    "patient_id": row["SubjectKeyRadiology"],
                    "exam_date": row["ExamDate"],
                    "arterial_image": a_img,
                    "portal_image": p_img,
                    "arterial_liver": a_liver,
                    "portal_liver": p_liver,
                    "arterial_lesion": a_lesion,
                    "portal_lesion": p_lesion
                }, save_path)

                counter += 1

            except Exception as e:
                print(f"Skipping {idx}: {e}", flush=True)
                skipped += 1

        return counter, skipped


def main():
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"], 0)

    data_dir = "/projects/net_contrast_classification/contrast_phase/Preprocessing/Segmentation_data/full_pairs.csv"
    dataset = pd.read_csv(data_dir)

    organ_ids = [5]

    preprocessor = SegPreprocess(
        dataset,
        organ_ids,
        pixdim=(1, 1, 1),
        # resize=(128, 128, 128)
    )

    splits = ["train", "val", "test", "inference"]
    split_name = splits[task_id % 4]

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