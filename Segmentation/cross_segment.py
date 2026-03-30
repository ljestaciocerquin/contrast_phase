import pandas as pd
import numpy as np
from pathlib import Path
import os, random, torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import PureWindowsPath
from monai.transforms import LoadImage
from monai.data import Dataset, DataLoader
from monai.transforms import (Compose,LoadImaged,ScaleIntensityd, ScaleIntensityRanged, Resized)
from monai.networks.nets import resnet10
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from monai.data import MetaTensor
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import segmentation_models_pytorch as smp

import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")

from Radiomics.models_pipe import train_tree_models
from Radiomics.data_preprocessing import radiomics_load, train_test_split, preprocess_train, preprocess_test
from Time_estim.time_estim import multi_channel, keep_organ_volume, get_2p5d_slices


def data_load(dataset, sample = None):

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
        labels = list(dataset.contrast)

    else:
        dataset['contrast_timing'] = (dataset["contrast"].str.cat(dataset["phase_timing"], sep=" "))
        dataset["contrast_timing"] = dataset["contrast_timing"].astype(str).str.strip()

        sampled_df = dataset.groupby('contrast_timing', group_keys=False).apply(lambda x: x.sample(n=min(len(x), sample), random_state=42))

        files = [os.path.join(row['server_folder'], PureWindowsPath(row['MatchKey']).name) for _, row in sampled_df.iterrows()]
        organ_files = [f.replace(".nii.gz", ".organs.nii.gz") for f in files]
        seg_files = [f.replace(".nii.gz", ".seg.nii.gz") for f in files]
        labels = list(sampled_df.contrast)



    return files, organ_files, seg_files, labels

class CTDataset(Dataset):
    def __init__(self, image_files, organ_files, seg_files, labels,
                 organ_ids, mode="3d", num_slices=None, transform=None):
        
        self.image_files = image_files
        self.organ_files = organ_files
        self.seg_files = seg_files
        self.labels = labels
        self.organ_ids = organ_ids
        self.mode = mode
        self.num_slices = num_slices
        self.transform = transform  # MONAI transforms

        self.loader = LoadImage(image_only=True, ensure_channel_first=True)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # -------- Load image --------
        image_array = self.loader(self.image_files[idx])
        organ_array = self.loader(self.organ_files[idx])
        seg_array = self.loader(self.seg_files[idx])
        label = self.labels[idx]

        organ_array = None
        volume = image_array  # default fallback

        # -------- Try organ processing --------
        try:
            organ_array_single = self.loader(self.organ_files[idx])

            # check empty explicitly (better than relying on exception)
            if np.size(organ_array_single) == 0:
                raise ValueError("Empty organ file")

            organ_array = multi_channel(self.organ_ids, organ_array_single)

            # crop ONLY if valid
            volume = keep_organ_volume(image_array, organ_array)

        except Exception as e:
            print(f"Fallback to full volume for {self.organ_files[idx]}: {e}")

        # -------- Mode handling --------
        if self.mode == "3d":
            x = volume

        elif self.mode == "2p5d":
            x = get_2p5d_slices(
                volume=volume,
                organ_array=organ_array,  # None if failed → handled inside
                num_slices=self.num_slices
            )

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # -------- Transforms --------
        if self.transform is not None:
            sample = {"image": x}
            sample = self.transform(sample)
            x = sample["image"]

        if not isinstance(x, MetaTensor):
            x = MetaTensor(x)

        y = seg_array
        if not isinstance(y, MetaTensor):
            y = MetaTensor(y)
        
        z = torch.tensor(label, dtype=torch.long)

        return {"image": x, "mask": y, "label":z}

def train_seg(model, train_loader, val_loader=None, epochs=10, lr=1e-4, weight_decay=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    bce_loss = torch.nn.BCEWithLogitsLoss()                                                     # binary loss for each pixel (lesion or not) => converts logits to probabilities using sigmoid; stabilizes learning 
    dice_loss = DiceLoss(sigmoid=True)                                                          # overlap loss (1-dice overlap); sigmoid=True => converts logits to probabilities; improves the segmentation quality

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    patience = 3
    counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # ------------------ TRAIN ------------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        dice_metric_train = DiceMetric(include_background=False, reduction="mean")
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)


        for batch in loop:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device).float()

            optimizer.zero_grad()
            outputs = model(images)                                      # output = logits, e.g., +3.2 → strong confidence lesion; -2.1 → strong confidence background; 0.0 → uncertain

            loss = dice_loss(outputs, masks) + bce_loss(outputs, masks)  # combine both losses because class imbalance is high for segmentation (areas without lesions >> areas with lesions)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            loop.set_postfix(loss=loss.item())

            preds = torch.sigmoid(outputs)                               # logits -> probabilities
            preds = (preds > 0.5).float()                                # probabilities -> binary segmentation mask

            train_correct += (preds == masks).sum().item()               # counts how many pixels are correctly predicted and accumulates this number for each batch
            train_total += masks.numel()                                 # accumulates the total number of pixels in the ground truth masks for each batch

            with torch.no_grad():  
                dice_metric_train(preds, masks)                          # accumulates results internally            


        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct/ train_total                       # pixel-wise accuracy (not segmentation!)
        train_dice = dice_metric_train.aggregate().item()                # mean dice over all batches - segmentation quality
        dice_metric_train.reset()                                        # reset for the next epoch


        # ---------------- VALIDATION ----------------
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            dice_metric = DiceMetric(include_background=False, reduction="mean")

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    masks = batch["mask"].to(device).float()

                    outputs = model(images)
                    loss = dice_loss(outputs, masks) + bce_loss(outputs, masks)
                    val_loss += loss.item()

                    # Dice score for validation
                    preds = torch.sigmoid(outputs)                               # logits -> probabilities
                    preds = (preds > 0.5).float()                                # probabilities -> binary segmentation mask

                    val_correct += (preds == masks).sum().item()               # counts how many pixels are correctly predicted and accumulates this number for each batch
                    val_total += masks.numel()                                 # accumulates the total number of pixels in the ground truth masks for each batch
                    dice_metric(preds, masks)                                  # accumulates results internally 

            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_correct / val_total               # pixel-wise accuracy
            val_dice = dice_metric.aggregate().item()           # mean dice over all batches - segmentation quality
            dice_metric.reset()                                 # reset for the next epoch

            print(
                f"Epoch {epoch+1}: "
                f"train_loss={avg_train_loss:.4f}, train_pixel_acc={avg_train_acc:.4f} , train_dice={train_dice:.4f} | "
                f"val_loss={avg_val_loss:.4f}, val_pixel_acc={avg_val_acc:.4f}, val_dice={val_dice:.4f}",
                flush=True
            )

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                best_model_state = model.state_dict()
                print("New best model")
            else:
                counter += 1
                print(f"No improvement ({counter}/{patience})")
                if counter >= patience:
                    print("Early stopping triggered")
                    break

    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

def evaluate_model(model, test_loader):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)                         # logits -> probabilities
            preds = (preds > 0.5).float()                          # probabilities -> binary segmentation mask

            correct += (preds == masks).sum().item()               # counts how many pixels are correctly predicted and accumulates this number for each batch
            total += masks.numel()                                 # accumulates the total number of pixels in the ground truth masks for each batch
            dice_metric(preds, masks)                              # accumulates results internally 

            avg_acc = correct / total                              # pixel-wise accuracy
            dice = dice_metric.aggregate().item()                  # mean dice over all batches - segmentation quality

    
    print(f"Test pixel-wise accuracy:{avg_acc:.4f} | Dice:{dice:.4f}")

    return preds

def create_loaders(data, val_size = 0.2, test_size = 0.2, mode = "3d"):

    files, organ_files, seg_files, labels_raw = data_load(data)

    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)

    # -------------------- Data split ----------------------

    temp_idx, test_idx = train_test_split(
    np.arange(len(files)),
    test_size=test_size,        # 20% test
    random_state=42,            # for reproducibility
    stratify=labels             # maintain class balance
    )

    train_idx, val_idx = train_test_split(
        temp_idx,
        test_size=val_size,
        random_state=42,
        stratify=labels[temp_idx]
    )

    # --------------- Transforms & Batches ------------------

    transforms_3d = Compose([ScaleIntensityRanged(keys=["image"],a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0,clip=True), 
                            Resized(keys=["image"], spatial_size=(64,64,64))])

    transforms_2d = Compose([ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
                            Resized(keys=["image"], spatial_size=(96, 96))])
    

    transforms, batches = [transforms_3d, 1 if mode == "3d" else transforms_2d, 32]


    # ---------------------- Train ------------------------

    train_dataset = CTDataset(
                        image_files = [files[i] for i in train_idx],
                        organ_files = [organ_files[i] for i in train_idx],
                        seg_files = [seg_files[i] for i in train_idx],
                        labels = labels[train_idx],
                        organ_ids=[5],                          # take only liver - we want to segment only the liver metastases => saves memory
                        transform=transforms
                    )
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batches,
        shuffle=False,
        num_workers=8,
        pin_memory=False
    )

    # -------------------- Validation --------------------

    val_dataset = CTDataset(
                            image_files = [files[i] for i in val_idx],
                            organ_files = [organ_files[i] for i in val_idx],
                            seg_files = [seg_files[i] for i in val_idx],
                            labels = labels[val_idx],
                            organ_ids=[5],                      
                            transform=transforms
                        )

    val_loader = DataLoader(
                            val_dataset,
                            batch_size=batches,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=False
                        )

    # ----------------------- Test --------------------

    test_dataset = CTDataset(
                            image_files = [files[i] for i in test_idx],
                            organ_files = [organ_files[i] for i in test_idx],
                            seg_files = [seg_files[i] for i in test_idx],
                            labels = labels[test_idx],
                            organ_ids=[5],                     
                            transform=transforms
                        )

    test_loader = DataLoader(
                            test_dataset,
                            batch_size=batches,
                            shuffle=False,
                            num_workers=0,                       # keep at zero
                            pin_memory=False
                        )


    return train_loader, val_loader, test_loader

def main():
    data = pd.read_csv("/projects/net_contrast_classification/contrast_phase/data/cleaned_data_1.csv")

    mode = "3d"

    train_loader, val_loader, test_loader = create_loaders(data, mode=mode)

    model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
    )

    trained_model = train_seg(model, train_loader, weights = None, val_loader=val_loader)

    pred_masks = evaluate_model(trained_model, test_loader)

if __name__ == "__main__":
    main()