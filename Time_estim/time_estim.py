import os, random, torch
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import PureWindowsPath
from monai.transforms import LoadImage
from monai.data import Dataset, DataLoader
from monai.transforms import (Compose,LoadImaged,ScaleIntensityd, ScaleIntensityRanged, Resized)
from monai.networks.nets import resnet10
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch.nn.functional as F
from monai.data import MetaTensor
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler

import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")

from Radiomics.radiomics_pipeline import list_organs, multi_channel

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def data_load(dataset, sample = None):
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
        labels = dataset.contrast_timing
    else:
        sampled_df = dataset.groupby('contrast_timing', group_keys=False).apply(lambda x: x.sample(n=min(len(x), sample), random_state=42))
        sampled_df['contrast_timing'] = sampled_df["contrast"].str.cat(sampled_df["phase_timing"], sep=" ")
        sampled_df["contrast_timing"] = sampled_df["contrast_timing"].astype(str).str.strip()

        files = [os.path.join(row['server_folder'], PureWindowsPath(row['MatchKey']).name) for i, row in sampled_df.iterrows()]
        organ_files = [f.replace(".nii.gz", ".organs.nii.gz") for f in files]
        labels = sampled_df.contrast_timing


    return files, organ_files, labels

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        """
        alpha: class weights (tensor of shape [num_classes]) or None
        gamma: focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        pt = torch.exp(-ce_loss)  # probability of correct class

        if self.alpha is not None:
            at = self.alpha[targets]
            ce_loss = at * ce_loss

        loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        
def keep_organ_volume(image_array, organ_array):
    mask = torch.tensor((organ_array > 0).any(axis=0), device=image_array.device)

    coords = mask.nonzero()

    h_min, w_min, d_min = coords.min(dim=0).values
    h_max, w_max, d_max = coords.max(dim=0).values

    ct_cropped = image_array[
        :,
        h_min:h_max+1,
        w_min:w_max+1,
        d_min:d_max+1
    ]

    mask_cropped = mask[h_min:h_max+1, w_min:w_max+1, d_min:d_max+1]

    ct_cropped[:, ~mask_cropped] = -1024

    return ct_cropped

def pad_to_shape(x, target_shape):
    """
    Pads a 3D image (C, H, W, D) to target_shape, centering the image in the padded volume.
    Pads with -1024 (air in CT).
    """
    _, H, W, D = x.shape
    _, Ht, Wt, Dt = target_shape

    pad_H = Ht - H
    pad_W = Wt - W
    pad_D = Dt - D

    # compute padding before and after for each dimension
    pad_H_before = pad_H // 2
    pad_H_after = pad_H - pad_H_before

    pad_W_before = pad_W // 2
    pad_W_after = pad_W - pad_W_before

    pad_D_before = pad_D // 2
    pad_D_after = pad_D - pad_D_before

    # F.pad expects padding in reverse order: (D, W, H)
    pad = (pad_D_before, pad_D_after,
           pad_W_before, pad_W_after,
           pad_H_before, pad_H_after)

    return F.pad(x, pad, value=-1024)  # use -1024 pading value because air ~ -1024 HU

def get_2p5d_slices(volume, organ_array = None, num_slices=7):
    """
    Extract a 2.5D stack of slices around the organ center.
    
    Args:
        volume: Tensor, shape (1, H, W, D)
        organ_array: Tensor or NumPy array, shape (C, H, W, D) or (H, W, D)
        num_slices: int, number of slices to extract
    
    Returns:
        slices: Tensor, shape (num_slices, H, W)
    """
    _, H, W, D = volume.shape

    if organ_array is not None:
        organ_tensor = torch.as_tensor(organ_array, device=volume.device)

        if organ_tensor.ndim > 3:
            organ_mask = (organ_tensor > 0).any(dim=0)  # (H, W, D)
        else:
            organ_mask = organ_tensor > 0

        # pick the slice with the most organ presence
        center = organ_mask.sum(dim=(0, 1)).argmax().item()

    else:
        center = volume.sum(dim=(1, 2)).argmax().item()

    half = num_slices // 2

    if num_slices % 2 == 0:
        indices = [min(max(center + i, 0), D - 1) for i in range(-half, half)]
    else:
        indices = [min(max(center + i, 0), D - 1) for i in range(-half, half + 1)]

    slices = volume[0, :, :, indices].permute(2, 0, 1)

    return slices

def create_3d_dataset(image_files, organ_files, organ_ids):
    loader = LoadImage(image_only=True, ensure_channel_first=True)
    
    cropped_images = []
    shapes = []
    valid_indices = []
    skipped = []

    for i, (file, organ_file) in enumerate(zip(image_files, organ_files)):
        try:
            image_array = loader(file)
            organ_array_single = loader(organ_file)

            organ_array = multi_channel(organ_ids, organ_array_single)
            ct_cropped = keep_organ_volume(image_array, organ_array)

            cropped_images.append(ct_cropped)
            shapes.append(ct_cropped.shape)
            valid_indices.append(i)

        except Exception as e:
            print(f"Skipping {organ_file}: {e}")
            skipped.append(organ_file)

    # Safety check
    if len(cropped_images) == 0:
        raise ValueError("No valid images found after loading.")

    # Compute mean shape
    shapes = np.array(shapes)
    mean_shape = np.round(shapes.mean(axis=0)).astype(int).tolist()
    print("Mean shape of cropped CT images:", mean_shape)
    print(f"Valid samples: {len(valid_indices)} | Skipped: {len(skipped)}")

    # Pad images to mean shape
    dataset = []

    for image in cropped_images:
        try:
            ct_padded = pad_to_shape(image, mean_shape)
            dataset.append(ct_padded)
        except Exception as e:
            print(f"Padding failed: {e}")



    return dataset, valid_indices

def train_cnn(model, train_loader, weights=None, val_loader=None, epochs=10, lr=1e-4, weight_decay = 1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if weights is not None:
        weights = weights.to(device)

    loss_fn = FocalLoss(alpha=weights, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float("inf")
    patience = 3
    counter = 0
    best_model_state = None

    for epoch in range(epochs):


        # -------------------- TRAIN --------------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch in loop:
            images = batch["image"].to(device)
            labels = batch["label"].long().to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)            # outputs = logits => pick the highest logit as the class label
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            loop.set_postfix(
                loss=loss.item(),
                acc=train_correct / train_total if train_total else 0.0
            )

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total else 0.0


        # -------------------- VALIDATION --------------------
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    labels = batch["label"].long().to(device)

                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = loss_fn(outputs, labels)

                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total if val_total else 0.0

            print(
                f"Epoch {epoch+1}: "
                f"train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f} | "
                f"val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f}",
                flush=True
            )

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                best_model_state = model.state_dict()  # saving best weights
                print("New best model")
            else:
                counter += 1
                print(f"No improvement ({counter}/{patience})")

                # if counter >= patience:
                #     print("Early stopping triggered")
                #     break

    # Loading the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

def evaluate_model(model, test_loader, device=None, class_names=None):
    """
    Evaluate a 3D CNN on a dataset.

    Args:
        model: Trained PyTorch model
        data_loader: DataLoader yielding {"image": tensor, "label": int}
        device: "cuda" or "cpu". If None, automatically uses CUDA if available
        class_names: list of class names for reporting

    Returns:
        metrics: dict with accuracy, classification report, and confusion matrix
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm
    }

    print(f"\nTest Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    return metrics

class Small3DCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
class CTDataset(Dataset):
    def __init__(self, image_files, organ_files, labels, organ_ids,
                 mode="3d", num_slices=5, transform=None):
        
        self.image_files = image_files
        self.organ_files = organ_files
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

        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return {"image": x, "label": y}

def create_loaders(data, organ_ids, val_size = 0.2, test_size = 0.2, mode = "3d", weights = False):

    files, organ_files, labels_raw = data_load(data)

    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)


    # --------------------- Data split -----------------------

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

    class_counts = np.bincount(labels[train_idx])
    weights = 1. / class_counts
    sample_weights = weights[labels[train_idx]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # --------------- Transforms & Batches ------------------

    transforms_3d = Compose([ScaleIntensityRanged(keys=["image"],a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0,clip=True), 
                            Resized(keys=["image"], spatial_size=(64,64,64))])

    transforms_2d = Compose([ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
                            Resized(keys=["image"], spatial_size=(96, 96))])
    

    transforms, batches = [transforms_3d, 1 if mode == "3d" else transforms_2d, 32]


    # ----------------------- Train -------------------------

    train_dataset = CTDataset(
                        image_files = [files[i] for i in train_idx],
                        organ_files = [organ_files[i] for i in train_idx],
                        labels = labels[train_idx],
                        organ_ids=organ_ids,                          # take only liver - we want to segment only the liver metastases => saves memory
                        transform=transforms
                    )
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batches,
        shuffle=False,
        num_workers=8,
        sampler=sampler,
        pin_memory=False
    )

    # -------------------- Validation --------------------

    val_dataset = CTDataset(
                            image_files = [files[i] for i in val_idx],
                            organ_files = [organ_files[i] for i in val_idx],
                            labels = labels[val_idx],
                            organ_ids=organ_ids,                      
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
                            labels = labels[test_idx],
                            organ_ids=organ_ids,                     
                            transform=transforms
                        )

    test_loader = DataLoader(
                            test_dataset,
                            batch_size=batches,
                            shuffle=False,
                            num_workers=0,                       # keep at zero
                            pin_memory=False
                        )

    if weights == True:
        return train_loader, val_loader, test_loader, le, weights
    else:
        return train_loader, val_loader, test_loader, le




def main():

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # ------------------------------------------------- Load data -------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------

    data = pd.read_csv("/projects/net_contrast_classification/contrast_phase/data/cleaned_data_1.csv")

    mode = "2p5d"
    num_slices = 7

    organ_ids = [1,2,3,5,8,9,13,51,52,63,64,65,66]
    
    
    train_loader, val_loader, test_loader, le, weights = create_loaders(data,organ_ids=organ_ids, mode=mode, weights=True)

    for batch in train_loader:
        print("Train batch:", batch["image"].shape)
        break

    for batch in val_loader:
        print("Validation batch:", batch["image"].shape)
        break

    # --------------------------------------------------- Train ---------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------


    # Initialize pretrained 3D or 2.5D ResNet model
    if mode == "3d":
        model = resnet10(spatial_dims=3, n_input_channels=1, num_classes=6)
    else:
        model = resnet10(spatial_dims=2, n_input_channels=num_slices, num_classes=6) # 2.5D pretrained CNN 
        in_features = model.fc.in_features

        # Add dropout for regularization
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 6)
            )


    # # Simple 3D CNN trained from scratch
    # model = Small3DCNN()

    trained_model = train_cnn(model,
                            train_loader,
                            weights = torch.tensor(weights, dtype=torch.float32),
                            val_loader=val_loader,
                            epochs = 20)

    # ------------------------------------------------- Evaluate --------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------
        
    # Saving the trained model
    save_path = "/projects/net_contrast_classification/contrast_phase/Time_estim/trained_model.pth"

    try:
        # ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(trained_model.state_dict(), save_path)
        print(f"Model saved successfully at: {save_path}")

    except Exception as e:
        print(f"Error saving model: {e}")

    
    # Evaluation 
    evaluate_model(trained_model, test_loader, class_names=le.classes_)


if __name__ == "__main__":
    main()