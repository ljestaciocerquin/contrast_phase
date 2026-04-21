import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import PureWindowsPath
import torchvision.models as models
from matplotlib.patches import Patch
from monai.data import Dataset, DataLoader
from monai.networks.nets import resnet10
from tqdm import tqdm
import torch.nn.functional as F
from typing import Optional

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class PairDataset(Dataset):
    def __init__(self, csv_path, split, sample=None, filter_outliers = False):
        self.df = pd.read_csv(csv_path)
        self.split = split

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if sample:
            self.df = self.df.iloc[:sample].reset_index(drop=True)

        self._check_paths()

        self.df["time_interval"] = self.df["time_interval"].astype("float32")

        if filter_outliers:
            self._filter_outliers(lower_bound=1, upper_bound=90)

    def _check_paths(self):
        exists_mask = self.df["output_path"].apply(os.path.exists)
        missing = self.df[~exists_mask]

        if len(missing) > 0:
            print(f"[WARNING] Missing files: {len(missing)}")
            print("Example:", missing["output_path"].head().tolist())

            self.df = self.df[exists_mask].reset_index(drop=True)
        else:
            print("All file paths exist")

    def _filter_outliers(self, lower_bound, upper_bound):
        mask = (
            (self.df["time_interval"] >= lower_bound) &
            (self.df["time_interval"] <= upper_bound)
        )

        self.df = self.df[mask].reset_index(drop=True)

        print(f"After filtering outliers: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        data = torch.load(row["output_path"], map_location="cpu")

        return {
            "patient_id": row["SubjectKeyRadiology"],
            "exam_date": row["ExamDate"],
            "arterial": torch.as_tensor(data["arterial_image"], dtype=torch.float32),
            "portal": torch.as_tensor(data["portal_image"], dtype=torch.float32),
            "time_interval": torch.tensor(row["time_interval"], dtype=torch.float32),
        }

class TimeEstimator(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        # Encoder 
        self.encoder = resnet10(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1  # dummy to keep structure valid => linear/classification layer [512, 1]
        )

        # Remove classification head
        self.encoder.fc = nn.Identity()         # the encoder now returns [B, 512, D, H, W]

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim * 3 + 1, 256),
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            # nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


    def encode(self, x):
        x = self.encoder(x)              # already [B, 512]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)    # safety fallback
        return x

    def forward(self, a_img, p_img):
        # initial shape of inputs = [B, 1, D, H, W]
        
        # encode both phases
        z_a = self.encode(a_img)            # [B, 512]
        z_p = self.encode(p_img)            # [B, 512]

        z_a = F.normalize(z_a, dim=1)
        z_p = F.normalize(z_p, dim=1)


        # feature interactions (important)
        z_diff = z_p - z_a                  # captures the changing contrast dynamics from arterial -> portal
        # z_mul = z_p * z_a                   # captures the correlation between different vessels' contrast - separate static vs dynamic features
        # z_abs = torch.abs(z_diff)           # stabilizes the training -> but we care about "direction"?
        # z_ratio = z_p / (z_a + 1e-6)

        z_cos = torch.sum(z_a * z_p, dim=1, keepdim=True)

        # concatenate everything
        z = torch.cat([z_a, z_p, z_diff, z_cos], dim=1)



        # predict time interval
        out = self.regressor(z)

        return out.squeeze(1)

def train_regressor(model, 
                    train_loader, 
                    val_loader = None, 
                    epochs=10,
                    lr=1e-4, 
                    weight_decay = 1e-4, 
                    early_stopping: Optional[int] = None):
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")

    counter = 0
    best_model_state = None

    train_curve = []
    val_curve = []


    # -------------------- TRAIN --------------------

    for epoch in range(epochs):

        train_loss = []
        model.train()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch in loop:
            a_imgs = batch["arterial"].to(device)
            p_imgs = batch["portal"].to(device)
            labels = batch["time_interval"].float().to(device)
            
            optimizer.zero_grad()

            outputs = model(a_imgs, p_imgs)
            error = torch.abs(outputs - labels)
            weights = torch.ones_like(labels)
            weights[(labels >= 30) & (labels <= 40)] = 3.0

            loss = (weights * error).mean()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())


        avg_train_loss = sum(train_loss) / len(train_loss)
        train_curve.append(avg_train_loss)
    
    # -------------------- VALIDATION --------------------

        if val_loader is not None:
            model.eval()
            val_loss =[]

            with torch.no_grad():
                for batch in val_loader:
                    a_imgs = batch["arterial"].to(device)
                    p_imgs = batch["portal"].to(device)
                    labels = batch["time_interval"].float().to(device)

                    outputs = model(a_imgs, p_imgs)

                    error = torch.abs(outputs - labels)
                    loss = error.mean()
                    
                    val_loss.append(loss.item())



            avg_val_loss = sum(val_loss) / len(val_loss)
            val_curve.append(avg_val_loss)

            print(
                    f"Epoch {epoch+1}: "
                    f"train_weighted_MAE={avg_train_loss:.4f}| val_MAE={avg_val_loss:.4f}",
                    flush=True)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                best_model_state = model.state_dict()  # saving best weights
                print(f"New best model: val_MAE = {best_val_loss}")
            else:
                counter += 1
                print(f"No improvement ({counter})")

                if early_stopping is not None and counter >= early_stopping:
                    print("Early stopping triggered")
                    break

    # Loading the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_curve, val_curve

def evaluate_regressor(model,
                       test_loader, 
                       device=None, 
                       threshold=None):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    sq_err_sum = 0.0
    abs_err_sum = 0.0
    n = 0

    results = []

    with torch.no_grad():
        for batch in test_loader:
            a_imgs = batch["arterial"].to(device)
            p_imgs = batch["portal"].to(device)
            labels = batch["time_interval"].float().to(device)

            patient_ids = batch["patient_id"]
            dates = batch["exam_date"]

            outputs = model(a_imgs, p_imgs)

            # errors
            error = outputs - labels
            abs_error = error.abs()

            # accumulate metrics
            sq_err_sum += (error ** 2).item()
            abs_err_sum += abs_error.item()
            n += 1

            err_value = abs_error.item()

            if threshold is None or err_value > threshold:
                results.append({
                    "patient_id": patient_ids[0],
                    "date": dates[0],
                    "true_interval": labels.item(),
                    "pred_interval": outputs.item(),
                    "error": err_value
                })

    mse = sq_err_sum / n
    mae = abs_err_sum / n
    rmse = mse ** 0.5

    print(f"Test MAE:  {mae:.4f} seconds")
    print(f"Test RMSE: {rmse:.4f} seconds")
    print(f"Test MSE:  {mse:.4f}")

    return results

def main():
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # ------------------------------------------------- Load data -------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------

    data_dir = "/projects/net_contrast_classification/contrast_phase/Preprocessing/Time_interval_data/paired_preprocessed_data.csv"

    batch_size = 1
    epochs = 20
    
    print(f"Training time interval estimation model with weights for extreme intervals and filtering", flush=True)

    train_dataset = PairDataset(data_dir, split = "train", filter_outliers=True)
    val_dataset = PairDataset(data_dir, split = "val", filter_outliers=True)
    test_dataset = PairDataset(data_dir, split = "test", filter_outliers=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TimeEstimator(embedding_dim=512)

    trained_model, train_curve, val_curve  = train_regressor(model,
                                                             train_loader,
                                                             val_loader,
                                                             epochs = epochs,
                                                             early_stopping=5)

    save_path = "/projects/net_contrast_classification/contrast_phase/DeepLClassifiers/Time_interval"
    os.makedirs(save_path, exist_ok=True)

    if train_curve is not None and val_curve is not None:
        plt.plot(train_curve, label="train")
        plt.plot(val_curve, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{save_path}/training_curve_filters.png") 


    # Saving the trained model
    save_path_model = f"{save_path}/trained_model_filters.pth"

    try:
        # ensure directory exists
        os.makedirs(os.path.dirname(save_path_model), exist_ok=True)

        torch.save(trained_model.state_dict(), save_path_model)
        print(f"Model saved successfully at: {save_path_model}", flush=True)

    except Exception as e:
        print(f"Error saving model: {e}", flush=True)

    # ------------------------------------------------- Evaluate --------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------

    test_losses = evaluate_regressor(trained_model, test_loader)
    df = pd.DataFrame(test_losses)
    df.to_csv(f"{save_path}/estimated_intervals_filters.csv", index=False)   

if __name__ == "__main__":
    main()
