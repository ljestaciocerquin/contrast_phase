import pandas as pd
import numpy as np
from pathlib import Path
import os, torch
from tqdm import tqdm
from pathlib import PureWindowsPath
from monai.data import Dataset, DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.metrics import MeanIoU
from monai.losses import DiceFocalLoss
import random

import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")
from Segmentation.cross_segment_models import LateFusionSegmentation, AttentionFusionSegmentation
from monai.transforms import Compose, RandCropByPosNegLabeld, EnsureTyped, MapTransform, RandSpatialCropd
from monai.inferers import sliding_window_inference
from monai.data import list_data_collate
import matplotlib.pyplot as plt


class ConditionalCropd(MapTransform):

    def __init__(self, keys, label_key, patch_size, num_patches=4):

        super().__init__(keys)
        self.label_key = label_key
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.pos_crop = RandCropByPosNegLabeld(
            keys=keys,
            label_key=label_key,
            spatial_size=patch_size,
            pos=1,
            neg=0,
            num_samples=num_patches
        )

        self.random_crop = RandSpatialCropd(
            keys=keys,
            roi_size=patch_size,
            random_size=False
        )

    def __call__(self, data):

        label = data[self.label_key]

        if torch.any(label > 0):
            return self.pos_crop(data)
        else:
            return [self.random_crop(data)]


class SegDataset(Dataset):
    def __init__(self, df, split, sample=None, masking=None, transform = None):
        self.df = df
        self.split = split
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.masking = masking
        self.transform = transform


        if sample:
            self.df = self.df.iloc[:sample].reset_index(drop=True)

        self._check_paths()
        self._filter_shape_mismatches()
        
        if split == "train":
            self.df = self.oversample_lesion_cases(self.df)
            print(f"Oversampled training set: {len(self.df)} samples")
    
    def _check_paths(self):
        exists_mask = self.df["output_path"].apply(os.path.exists)
        missing = self.df[~exists_mask]

        if len(missing) > 0:
            print(f"[WARNING] Missing files: {len(missing)}")
            print("Example:", missing["output_path"].head().tolist())

            self.df = self.df[exists_mask].reset_index(drop=True)
        else:
            print("All file paths exist")

    def _filter_shape_mismatches(self):
        keep = []

        for _, row in self.df.iterrows():
            data = torch.load(row["output_path"], map_location="cpu")
            tensors = [
                data["arterial_image"],
                data["portal_image"],
                data["arterial_liver"],
                data["portal_liver"],
                data["arterial_lesion"],
                data["portal_lesion"],
            ]

            # Skip samples with missing tensors
            if any(t is None for t in tensors):
                keep.append(False)
                continue

            # All tensors must have the same shape
            shapes = [tuple(t.shape) for t in tensors]
            keep.append(len(set(shapes)) == 1)

        removed = len(keep) - sum(keep)
        print(f"Removed {removed} samples with missing or mismatched tensor shapes.")

        self.df = self.df[keep].reset_index(drop=True)

    def apply_masking(self, image, organ_mask, lesion_mask):
        if self.masking == "concat":
            return torch.cat([image, organ_mask, lesion_mask], dim=0)
        elif self.masking == "crop":
            combined_mask = ((organ_mask + lesion_mask) > 0).float()            # mask that takes the liver + lesion region for cases when the lesion goes far outside of the liver
            return image * combined_mask
        elif self.masking is None:
            return image
        else:
            raise ValueError("Invalid masking option")

    def oversample_lesion_cases(self, df):

        lesion_df = df[
            (df["arterial_lesionfree"] == "No") |
            (df["portal_lesionfree"] == "No")
        ]

        negative_df = df[
            (df["arterial_lesionfree"] == "Yes") &
            (df["portal_lesionfree"] == "Yes")
        ]

        df_balanced = pd.concat([
            negative_df,
            lesion_df,
            lesion_df,
            lesion_df
        ])

        return df_balanced.sample(frac=1).reset_index(drop=True)

        
    def ground_truth(self, a_lesion, p_lesion): 

        if a_lesion is None and p_lesion is None:
            raise ValueError("Both lesion masks are missing")

        if a_lesion is None:
            return (p_lesion > 0).float()

        if p_lesion is None:
            return (a_lesion > 0).float()

        return torch.logical_or(a_lesion > 0, p_lesion > 0).float()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = torch.load(row["output_path"])



        if data["arterial_lesion"] is None or data["portal_lesion"] is None:
            raise RuntimeError(
                f"Bad saved sample:\n"
                f"path={row['output_path']}\n"
                f"a_lesion={data['arterial_lesion'] is None}\n"
                f"p_lesion={data['portal_lesion'] is None}"
            )

        a_image, p_image = data["arterial_image"], data["portal_image"]
        a_liver, p_liver = data["arterial_liver"], data["portal_liver"]
        a_lesion, p_lesion = data["arterial_lesion"], data["portal_lesion"]

        gt_mask = self.ground_truth(a_lesion, p_lesion)  

        a_image = self.apply_masking(a_image, a_liver, a_lesion)
        p_image = self.apply_masking(p_image, p_liver, p_lesion)



        # stack modalities as channels
        image = torch.cat([a_image, p_image], dim=0)
        sample = {
            "image": image,
            "label": gt_mask

        }

        if self.transform:
            sample = self.transform(sample)


        return sample


def modality_dropout(ap, pvp, p=0.3):
    """
    Modality dropout to force the model to learn from single modality input for inference 
    """
    drop_ap = random.random() < p
    drop_pvp = random.random() < p

    # prevent both being dropped => randomly and with equal probability keep one of the two
    if drop_ap and drop_pvp:
        if random.random() < 0.5:
            drop_ap = False
        else:
            drop_pvp = False

    if drop_ap:
        ap = None
    if drop_pvp:
        pvp = None

    return ap, pvp

def plot_curve(train_curve, val_curve, save_path, metric = "Loss"):
    plt.clf()  
    plt.plot(train_curve, label=f"Train {metric}")
    plt.plot(val_curve, label=f"Validation {metric}")
    plt.title(f"{metric} curve")
    plt.xlabel("Epoch")
    plt.ylabel(f"{metric}")
    plt.legend()
    plt.savefig(save_path)


def train_seg(model,
              train_loader,
              val_loader, 
              epochs=10, 
              lr=1e-4, 
              weight_decay=1e-4,
              bce_weight=0.5,
              dice_weight=0.5,
              early_stopping=None,
              device=None,
              save_path=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    bce_loss = torch.nn.BCEWithLogitsLoss()                                                     # binary loss for each pixel (lesion or not) => converts logits to probabilities using sigmoid; stabilizes learning 
    dice_loss = DiceLoss(sigmoid=True, squared_pred=True, smooth_nr=1e-5, smooth_dr=1e-5)       # overlap loss (1-dice overlap); sigmoid=True => converts logits to probabilities; improves the segmentation quality
    dice_focalloss = DiceFocalLoss(
    sigmoid=True,
    lambda_dice=1.0,
    lambda_focal=1.0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    counter = 0
    best_model_state = None

    all_means = []
    all_stds = []

    train_dice_curve = []
    train_loss_curve = []
    val_dice_curve = []
    val_loss_curve = []


    for epoch in range(epochs):
        # ------------------ TRAIN ------------------
        model.train()
        train_loss = 0.0
        train_dice_loss = 0.0
        train_bce = 0.0

        dice_metric_train = DiceMetric(include_background=False, reduction="mean")
        iou_metric_train = MeanIoU(include_background=False)
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)


        for batch in loop:

            image = batch["image"].to(device)
            lesion = batch["label"].to(device).float()
            a_img = image[:, 0:1]   # arterial
            p_img = image[:, 1:2]   # portal


            if isinstance(model, AttentionFusionSegmentation):
                a_img, p_img = modality_dropout(a_img, p_img)


            optimizer.zero_grad()
            outputs = model(a_img, p_img)                                      # output = logits, e.g., +3.2 → strong confidence lesion; -2.1 → strong confidence background; 0.0 → uncertain

            all_means.append(outputs.mean().item())
            all_stds.append(outputs.std().item())

            dice = dice_loss(outputs, lesion)
            dice_focal = dice_focalloss(outputs, lesion)
            bce = bce_loss(outputs, lesion)
            loss = (dice_weight * dice_focal + bce_weight * bce)   # combine both losses because class imbalance is high for segmentation (areas without lesions >> areas with lesions)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice_loss += dice.item()
            train_bce += bce.item()


            loop.set_postfix(loss=loss.item())

            probs = torch.sigmoid(outputs)                                 # logits -> probabilities
            preds = (probs > 0.3).float()                                # probabilities -> binary segmentation mask

            with torch.no_grad():  
                dice_metric_train(preds, lesion)                            # accumulates results internally  
                iou_metric_train(preds, lesion)  


        # Investigate logits mean and standard deviations
        epoch_mean = sum(all_means) / len(all_means)
        epoch_std = sum(all_stds) / len(all_stds)


        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice_loss = train_dice_loss / len(train_loader)
        avg_train_bce = train_bce / len(train_loader)

        train_dice = dice_metric_train.aggregate().item()                # mean dice over all batches - segmentation quality
        dice_metric_train.reset()                                        # reset for the next epoch
        train_iou = iou_metric_train.aggregate().item()
        iou_metric_train.reset()

        train_dice_curve.append(train_dice)
        train_loss_curve.append(avg_train_loss)


        # ---------------- VALIDATION ----------------
        if val_loader is not None:
            model.eval()
            val_loss = 0.0


            dice_metric = DiceMetric(include_background=False, reduction="mean")
            iou_metric = MeanIoU(include_background=False)

            with torch.no_grad():
                for batch in val_loader:
                    image = batch["image"].to(device)
                    lesion = batch["label"].to(device).float()
                    a_img = image[:, 0:1]   # arterial
                    p_img = image[:, 1:2]   # portal
                    
                    if isinstance(model, AttentionFusionSegmentation):
                        a_img, p_img = modality_dropout(a_img, p_img)


                    outputs = model(a_img, p_img)
                    dice = dice_loss(outputs, lesion)
                    dice_focal = dice_focalloss(outputs, lesion)
                    bce = bce_loss(outputs, lesion)

                    loss = (dice_weight * dice_focal + bce_weight * bce)   
                    val_loss += loss.item()

                    # Dice score for validation
                    probs = torch.sigmoid(outputs)                               # logits -> probabilities
                    preds = (probs > 0.3).float()                                # probabilities -> binary segmentation mask

                    dice_metric(preds, lesion)                                  # accumulates results internally 
                    iou_metric(preds, lesion)



            avg_val_loss = val_loss / len(val_loader)
            val_dice = dice_metric.aggregate().item()           # mean dice over all batches - segmentation quality
            dice_metric.reset()                                 # reset for the next epoch
            val_iou = iou_metric.aggregate().item()
            iou_metric.reset()

            val_dice_curve.append(val_dice)
            val_loss_curve.append(avg_val_loss)

            print(
            f"""
            Epoch {epoch+1}
            ---------------------------------------------
            Logits mean: {epoch_mean:.4f}, Logits std: {epoch_std:.4f}

            train_loss      : {avg_train_loss:.4f}
            train_bce       : {avg_train_bce:.4f}
            train_dice_loss : {avg_train_dice_loss:.4f}
            train_dice      : {train_dice:.4f}
            train_iou       : {train_iou:.4f}

            val_loss        : {avg_val_loss:.4f}
            val_dice        : {val_dice:.4f}
            val_iou         : {val_iou:.4f}
            ---------------------------------------------
            """
            )

            if train_loss_curve is not None and val_loss_curve is not None:
                plot_curve(train_loss_curve, val_loss_curve, title="Loss Curve", save_path=f"{save_path}/loss_{model.__class__.__name__}.png", metric="Loss")
            if train_dice_curve is not None and val_dice_curve is not None:
                plot_curve(train_dice_curve, val_dice_curve, title="Dice Curve", save_path=f"{save_path}/dice_{model.__class__.__name__}.png", metric="Dice")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                best_model_state = model.state_dict()
                torch.save(best_model_state, f"{save_path}/{model.__class__.__name__}.pth")
                print("New best model saved")
            else:
                counter += 1
                if early_stopping:
                    patience = early_stopping
                    print(f"No improvement ({counter}/{patience})")
                    if counter >= patience:
                        print("Early stopping triggered")
                        break
                else:
                    print(f"No improvement ({counter})")

    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

def evaluate_model(model, test_loader, patch_size = (64, 64, 64)):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)
            lesion = batch["label"].to(device)
            # sliding window expects full volume input

            def predictor(x_in):
                a = x_in[:, 0:1]
                p = x_in[:, 1:2]
                return model(a, p)

            out = sliding_window_inference(
                inputs=image,
                roi_size=patch_size,
                sw_batch_size=4,
                predictor=predictor,
                overlap=0.5,
                mode="gaussian",
            )

            preds = (torch.sigmoid(out) > 0.5).float()
            dice_metric(preds, lesion)

    print(f"Test Dice: {dice_metric.aggregate().item():.4f}")
    return preds

def main():
    data_dir = "/projects/net_contrast_classification/contrast_phase/Preprocessing/Segmentation_data/full_pairs.csv"
    df = pd.read_csv(data_dir)
    df = df[df["split"] != "inference"]
    
    # batch_size = 32
    epochs = 100
    dice_weight = 0.8
    bce_weight = 0.2
    patch_size = (64, 64, 64)

    transform = Compose([
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=3,        # 3 positive patches
            neg=1,        # 1 negative patch
            num_samples=4,
            image_key="image"
        ),

        EnsureTyped(keys=["image", "label"])

    ])

    # transform = Compose([
    #     ConditionalCropd(
    #         keys=["image", "label"],
    #         label_key="label",
    #         patch_size=patch_size,
    #         num_patches=4
    #     ),
    #     EnsureTyped(keys=["image", "label"])

    # ])

    train_dataset = SegDataset(df, split = "train", transform=transform)
    val_dataset = SegDataset(df, split = "val", transform=transform)
    test_dataset = SegDataset(df, split = "test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_map = {0: "Late", 
                 1: "Attention",
                }

    
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    model_name = model_map[task_id]

    model = LateFusionSegmentation() if model_name == "Late" else AttentionFusionSegmentation()

    print("=" * 80, flush=True) 
    print(f"Training cross-modality segmentation model: {model.__class__.__name__}", flush=True)
    print(f"Dice weight: {dice_weight}, BCE weight: {bce_weight}, Epochs: {epochs}\n", flush=True)
    print(f"Number of training samples: {len(train_dataset)}", flush=True)
    print(f"Number of validation samples: {len(val_dataset)}", flush=True)
    print(f"Number of test samples: {len(test_dataset)}", flush=True)
    print("=" * 80, flush=True)
    print("\n", flush=True) 

    save_path = f"/projects/net_contrast_classification/contrast_phase/Segmentation"

    trained_model = train_seg(model, 
                              train_loader, 
                              val_loader,
                              epochs=epochs,
                              bce_weight=bce_weight,
                              dice_weight=dice_weight,
                              early_stopping=None,
                              save_path = save_path
                              )
    

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(trained_model.state_dict(), save_path)
        print(f"Model saved successfully at: {save_path}", flush=True)

    except Exception as e:
        print(f"Error saving model: {e}", flush=True)


    preds = evaluate_model(trained_model, test_loader, patch_size=patch_size)

if __name__ == "__main__":
    main()