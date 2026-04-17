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
from monai.networks.nets import resnet10, resnet18
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class PairDataset(Dataset):
    def __init__(self, folder, sample=None):
        self.files = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".pt")
        ])

        if sample:
            self.files = self.files[:sample]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")

        patient_id = data["patient_id"]
        date = data["exam_date"]
        a_img = data["arterial_image"]
        p_img = data["portal_image"]
        time_interval = data["time_interval"]

        return {
            "patient_id": patient_id,   
            "exam_date": date,                       
            "arterial": a_img,
            "portal": p_img,
            "time_interval": time_interval
            }

class TimeEstimator(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        # Encoder 
        self.encoder = resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1  # dummy to keep structure valid => linear/classification layer [512, 1]
        )

        # Remove classification head
        self.encoder.fc = nn.Identity()         # the encoder now returns [B, 512, D, H, W]

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim * 4, 256),
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


        # feature interactions (important)
        z_diff = z_p - z_a                  # captures the changing contrast dynamics from arterial -> portal
        z_mul = z_p * z_a                   # captures the correlation between different vessels' contrast - separate static vs dynamic features
        # z_abs = torch.abs(z_diff)           # stabilizes the training -> but we care about "direction"?
        # z_ratio = z_p / (z_a + 1e-6)

        # concatenate everything
        z = torch.cat([z_a, z_p, z_diff, z_mul
                    #    , z_abs, z_ratio
                       ], dim=1)



        # predict time interval
        out = self.regressor(z)

        return out.squeeze(1)

def train_regressor(model, train_loader, val_loader = None, epochs=10, lr=1e-4, weight_decay = 1e-4, early_stopping = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    loss_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    patience = 3
    counter = 0
    best_model_state = None

    train_curve = []
    val_curve = []


    # -------------------- TRAIN --------------------

    for epoch in range(epochs):
        train_losses = []
        train_mae = []

        model.train()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch in loop:
            a_imgs = batch["arterial"].to(device)
            p_imgs = batch["portal"].to(device)
            labels = batch["time_interval"].float().to(device)
            
            optimizer.zero_grad()

            outputs = model(a_imgs, p_imgs)
            mse = loss_fn(outputs, labels)
            mae = mae_fn(outputs, labels)

            mse.backward()
            optimizer.step()

            train_losses.append(mse.item())
            train_mae.append(mae.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_mae = sum(train_mae) / len(train_mae)
        train_curve.append(avg_train_loss)
    
    # -------------------- VALIDATION --------------------

        if val_loader is not None:
            model.eval()
            val_losses = []
            val_mae = []

            with torch.no_grad():
                for batch in val_loader:
                    a_imgs = batch["arterial"].to(device)
                    p_imgs = batch["portal"].to(device)
                    labels = batch["time_interval"].float().to(device)

                    outputs = model(a_imgs, p_imgs)

                    mse = loss_fn(outputs, labels)
                    mae = mae_fn(outputs, labels)

                    val_losses.append(mse.item())
                    val_mae.append(mae.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_mae = sum(val_mae) / len(val_mae)
            val_curve.append(avg_val_loss)

            print(
                    f"Epoch {epoch+1}: "
                    f"train_MSE={avg_train_loss:.4f}, train_MAE={avg_train_mae:.4f}| val_MSE={avg_val_loss:.4f}, val_MAE={avg_val_mae:.4f}",
                    flush=True)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                best_model_state = model.state_dict()  # saving best weights
                print(f"New best model: val_loss = {best_val_loss}")
            else:
                counter += 1
                print(f"No improvement ({counter}/{patience})")
                if early_stopping:
                    if counter >= patience:
                        print("Early stopping triggered")
                        break

    # Loading the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    plt.plot(train_curve, label="train_loss")
    if val_loader is not None:
        plt.plot(val_curve, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    return model

def evaluate_regressor(model, test_loader, device=None, threshold=None):
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

    path = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/pairs_preprocessed"
    batch_size = 1
    epochs = 10
    
    print(f"Training time interval estimation model", flush=True)

    train_dataset = PairDataset(f"{path}/train")
    val_dataset = PairDataset(f"{path}/val")
    test_dataset = PairDataset(f"{path}/test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TimeEstimator(embedding_dim=512)

    trained_model = train_regressor(model, train_loader, val_loader, epochs = epochs)

    # ------------------------------------------------- Evaluate --------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------
        
    # Saving the trained model
    save_path = f"/projects/net_contrast_classification/contrast_phase/DeepLClassifiers/time_estim_trained_model.pth"

    try:
        # ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(trained_model.state_dict(), save_path)
        print(f"Model saved successfully at: {save_path}", flush=True)

    except Exception as e:
        print(f"Error saving model: {e}", flush=True)

    test_losses = evaluate_regressor(trained_model, test_loader)
    df = pd.DataFrame(test_losses)
    df.to_csv(f"/projects/net_contrast_classification/contrast_phase/DeepLClassifiers/estimated_intervals.csv", index=False)   


if __name__ == "__main__":
    main()
