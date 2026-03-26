import os, random, torch
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import PureWindowsPath
# import torchvision.models as models
# from matplotlib.patches import Patch
# from monai.transforms import LoadImage
from monai.data import Dataset, DataLoader
from monai.transforms import (Compose,LoadImaged,ScaleIntensityd, ScaleIntensityRanged, Resized)
from monai.networks.nets import resnet10, resnet18
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
# import torch.nn.functional as F
from monai.data import MetaTensor
from sklearn.utils.class_weight import compute_class_weight
from time_estim import keep_organ_volume, get_2p5d_slices, train_cnn, FocalLoss

import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")
# print(sys.path)
from Radiomics.radiomics_pipeline import list_organs, multi_channel

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# CT slices → ResNet → feature vectors → LSTM → time prediction

# class CTDataset(Dataset):
#     def __init__(self, image_files, organ_files, times, organ_ids,
#                  mode="3d", num_slices=5, transform=None):
        
#         self.image_files = image_files
#         self.organ_files = organ_files
#         self.times = times
#         self.organ_ids = organ_ids
#         self.mode = mode
#         self.num_slices = num_slices
#         self.transform = transform  # MONAI transforms

#         self.loader = LoadImage(image_only=True, ensure_channel_first=True)

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx, retries = 3):
#         try:
#             # ---------------- Load ----------------
#             image_array = self.loader(self.image_files[idx])
#             organ_array_single = self.loader(self.organ_files[idx])

#             # ---------------- Multi-channel mask ----------------
#             organ_array = multi_channel(self.organ_ids, organ_array_single)

#             # ---------------- Crop ----------------
#             ct_cropped = keep_organ_volume(image_array, organ_array)

#             # ---------------- Mode handling ----------------
#             if self.mode == "3d":
#                 x = ct_cropped  # (1, H, W, D)
#             elif self.mode == "2p5d":
#                 x = get_2p5d_slices(ct_cropped, organ_array, num_slices=self.num_slices)  # (k, H, W)
#             else:
#                 raise ValueError(f"Unknown mode: {self.mode}")

#             # ---------------- Apply transforms ----------------
#             if self.transform is not None:
#                 sample = {"image": x}
#                 sample = self.transform(sample)
#                 x = sample["image"]

#             # ---------------- Ensure MONAI MetaTensor ----------------
#             if not isinstance(x, MetaTensor):
#                 x = MetaTensor(x)

#             y = torch.tensor(self.times[idx], dtype=torch.float32)

#             return {"image": x, "times": y}
        
#         # ---------------- Skip faulty files ----------------
#         except Exception as e:
#             print(f"Skipping {self.organ_files[idx]}: {e}")

#             if retries <= 0:
#                 raise RuntimeError(f"Too many failed retries at index {idx}")

#             return self.__getitem__((idx + 1) % len(self), retries=retries - 1)
        
# def train_cnn(model, train_loader, weights=None, val_loader=None, epochs=10, lr=1e-4, weight_decay = 1e-5):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = model.to(device)

#     if weights is not None:
#         weights = weights.to(device)

#     loss_fn = torch.nn.SmoothL1Loss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scaler = torch.cuda.amp.GradScaler()

#     for epoch in range(epochs):


#         # -------------------- TRAIN --------------------
#         model.train()
#         train_loss = 0.0
#         train_correct = 0
#         train_total = 0

#         loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

#         for batch in loop:
#             images = batch["image"].to(device)
#             labels = batch["times"].long().to(device)

#             optimizer.zero_grad()

#             with torch.cuda.amp.autocast():
#                 outputs = model(images)
#                 loss = loss_fn(outputs, labels)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             train_loss += loss.item()
#             preds = torch.argmax(outputs, dim=1)
#             train_correct += (preds == labels).sum().item()
#             train_total += labels.size(0)

#             loop.set_postfix(
#                 loss=loss.item(),
#                 acc=train_correct / train_total if train_total else 0.0
#             )

#         avg_train_loss = train_loss / len(train_loader)
#         train_acc = train_correct / train_total if train_total else 0.0


#         # -------------------- VALIDATION --------------------
#         if val_loader is not None:
#             model.eval()
#             val_loss = 0.0
#             val_correct = 0
#             val_total = 0

#             with torch.no_grad():
#                 for batch in val_loader:
#                     images = batch["image"].to(device)
#                     labels = batch["times"].long().to(device)

#                     with torch.cuda.amp.autocast():
#                         outputs = model(images)
#                         loss = loss_fn(outputs, labels)

#                     val_loss += loss.item()
#                     preds = torch.argmax(outputs, dim=1)
#                     val_correct += (preds == labels).sum().item()
#                     val_total += labels.size(0)

#             avg_val_loss = val_loss / len(val_loader)
#             val_acc = val_correct / val_total if val_total else 0.0

#             print(
#                 f"Epoch {epoch+1}: "
#                 f"train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f} | "
#                 f"val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f}",
#                 flush=True
#             )

#         else:
#             print(
#                 f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f}",
#                 flush=True
#             )

#     return model

# def evaluate_model(model, test_loader, device=None, class_names=None):
#     """
#     Evaluate a 3D CNN on a dataset.

#     Args:
#         model: Trained PyTorch model
#         data_loader: DataLoader yielding {"image": tensor, "label": int}
#         device: "cuda" or "cpu". If None, automatically uses CUDA if available
#         class_names: list of class names for reporting

#     Returns:
#         metrics: dict with accuracy, classification report, and confusion matrix
#     """
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     model.to(device)
#     model.eval()

#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in test_loader:
#             images = batch["image"].to(device)
#             labels = batch["times"].to(device)

#             outputs = model(images)
#             preds = torch.argmax(outputs, dim=1)

#             all_preds.append(preds.cpu())
#             all_labels.append(labels.cpu())

#     all_preds = torch.cat(all_preds)
#     all_labels = torch.cat(all_labels)

#     acc = accuracy_score(all_labels, all_preds)

#     metrics = {
#         "accuracy": acc
#     }

#     print(f"\nTest Accuracy: {acc:.4f}")

#     return metrics


# def main():
#     # ------------------------------------------------- Load data -------------------------------------------------
#     # -------------------------------------------------------------------------------------------------------------

#     data = pd.read_csv("/projects/net_contrast_classification/contrast_phase/data/cleaned_data_1.csv")
#     folder_path = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET"

#     data = data[(data.contrast.isin(['Arterial', "Portal"])) 
#                 & (data.exist_on_server.notna())
#                 & (data.is_liver_imaged == "Yes")
#                 & (data.phase_timing != '0.0')]

#     # data['contrast_timing'] = (data["contrast"].str.cat(data["phase_timing"], sep=" "))
#     # data["contrast_timing"] = data["contrast_timing"].astype(str).str.strip()

#     sampled_df = data.groupby('contrast_timing', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 600), random_state=42))
#     sampled_df['contrast_timing'] = sampled_df["contrast"].str.cat(sampled_df["phase_timing"], sep=" ")
#     sampled_df["contrast_timing"] = sampled_df["contrast_timing"].astype(str).str.strip()

#     files = [os.path.join(folder_path, PureWindowsPath(f).name) for f in sampled_df["MatchKey"]]
#     organ_files = [os.path.join(folder_path, PureWindowsPath(os.path.basename(f).replace(".nii.gz", ".organs.nii.gz")).name) for f in files]

#     organ_ids,_ = list_organs(selected = [1,2,3,5,8
#                                           ,9,13,51,52
#                                           ,63,64,65,66
#                                           ], by = "id")

#     # le = LabelEncoder()
#     mode = "2p5d"
#     num_slices = 5

#     times = sampled_df.AcquisitionTime

#     temp_idx, test_idx = train_test_split(
#         np.arange(len(files)),
#         test_size=0.2,        # 20% test
#         random_state=42,      # for reproducibility
#         stratify=times       # maintain class balance
#     )


#     train_idx, val_idx = train_test_split(
#         temp_idx,
#         test_size=0.2,
#         random_state=42,
#         stratify=times[temp_idx]
#     )

#     transforms_3d = Compose([ScaleIntensityRanged(keys=["image"],a_min=-1000, a_max=1000,
#                                                b_min=0.0, b_max=1.0,clip=True), 
#                             Resized(keys=["image"], spatial_size=(64,64,64))])
    
#     transforms_2d = Compose([ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000,
#                                                 b_min=0.0, b_max=1.0, clip=True),
#                             Resized(keys=["image"], spatial_size=(96, 96))])




#     # --------------------------------------------------- Train ---------------------------------------------------
#     # -------------------------------------------------------------------------------------------------------------

#     if mode == "3d":
#         transforms, batches = transforms_3d, 1
        
#     else:
#         transforms, batches = transforms_2d, 16


#     # Train dataset and dataloader
#     train_dataset = CTDataset(
#         [files[i] for i in train_idx],
#         [organ_files[i] for i in train_idx],
#         times[train_idx],
#         organ_ids,
#         mode=mode,
#         num_slices=num_slices,
#         transform=transforms
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batches,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=False
#     )

#     # Validation dataset and dataloader
#     val_dataset = CTDataset(
#         [files[i] for i in val_idx],
#         [organ_files[i] for i in val_idx],
#         times[val_idx],
#         organ_ids,
#         mode=mode,
#         num_slices=num_slices,
#         transform=transforms
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batches,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=False
#     )
    
#     for batch in train_loader:
#         print("Train batch:", batch["image"].shape)
#         break

#     for batch in val_loader:
#         print("Validation batch:", batch["image"].shape)
#         break


#     # Initialize pretrained 3D or 2.5D ResNet model
#     if mode == "3d":
#         model = resnet10(spatial_dims=3, n_input_channels=1, num_classes=1)
#     else:
#         model = resnet10(spatial_dims=2, n_input_channels=num_slices, num_classes=1) # 2.5D pretrained CNN 


#     # # Simple 3D CNN trained from scratch
#     # model = Small3DCNN()

#     # train_labels = times[train_idx]
#     # weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels),y=train_labels)
#     # weights = torch.tensor(weights, dtype=torch.float32)

#     trained_model = train_cnn(model, train_loader, weights = None, val_loader=val_loader)



#     # ------------------------------------------------- Evaluate --------------------------------------------------
#     # -------------------------------------------------------------------------------------------------------------

#     test_dataset = CTDataset(
#         [files[i] for i in test_idx],
#         [organ_files[i] for i in test_idx],
#         times[test_idx],
#         organ_ids,
#         mode=mode,
#         num_slices=num_slices,
#         transform=transforms
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batches,
#         shuffle=True,
#         num_workers=0,
#         pin_memory=False
#         )
        
#     evaluate_model(trained_model, test_loader)


# if __name__ == "__main__":
#     main()
