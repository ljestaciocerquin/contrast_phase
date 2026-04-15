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
        self.encoder = resnet10(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1  # dummy to keep structure valid => linear/classification layer [512, 1]
        )

        # Remove classification head
        self.encoder.fc = nn.Identity()         # the encoder now returns [B, 512, D, H, W]

        # Pooling to get fixed-size embedding
        self.pool = nn.AdaptiveAvgPool3d(1)     # [B, 512, 1, 1, 1]

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def encode(self, x):
        """
        Encode input volume into embedding vector
        """
        x = self.encoder(x)                 # [B, 512, D, H, W]
        x = self.pool(x)                    # [B, 512, 1, 1, 1]
        x = x.view(x.size(0), -1)           # [B, 512]
        return x
    

    def forward(self, a_img, p_img):
        # initial shape of inputs = [B, 1, D, H, W]
        
        # encode both phases
        z_a = self.encode(a_img)            # [B, 512]
        z_p = self.encode(p_img)            # [B, 512]

        print("z_a:", z_a.shape)
        print("z_p:", z_p.shape)

        # feature interactions (important)
        z_diff = z_p - z_a                  # captures the changing contrast dynamics from arterial -> portal
        z_mul = z_p * z_a                   # captures the correlation between different vessels' contrast - separate static vs dynamic features
        # z_abs = torch.abs(z_diff)         # stabilizes the training -> but we care about "direction"?

        # concatenate everything
        z = torch.cat([z_a, z_p, z_diff, z_mul], dim=1)
        print("z:", z.shape)

        # predict time interval
        out = self.regressor(z)

        return out.squeeze(1)

def train_regressor(model, train_loader, val_loader = None, epochs=10, lr=1e-4, weight_decay = 1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    patience = 3
    counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    train_curve = []
    val_curve = []


    # -------------------- TRAIN --------------------

    for epoch in range(epochs):

        model.train()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch in loop:
            a_imgs = batch["arterial"].to(device)
            p_imgs = batch["portal"].to(device)
            labels = batch["time_interval"].float().to(device)
            
            optimizer.zero_grad()

            outputs = model(a_imgs, p_imgs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        train_curve.append(avg_train_loss)
    
    # -------------------- VALIDATION --------------------

        if val_loader is not None:
            model.eval()

            with torch.no_grad():
                for batch in val_loader:
                    a_imgs = batch["arterial"].to(device)
                    p_imgs = batch["portal"].to(device)
                    labels = batch["time_interval"].float().to(device)

                    outputs = model(a_imgs, p_imgs)
                    loss = loss_fn(outputs, labels)

                    val_losses.append(loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            val_curve.append(avg_val_loss)

            print(
                    f"Epoch {epoch+1}: "
                    f"train_loss={avg_train_loss:.4f}| val_loss={avg_val_loss:.4f}",
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
    
    model.to(device)
    model.eval()

    loss_fn = nn.MSELoss()
    test_losses = []
    all_errors = []
    results = []

    with torch.no_grad():
        for batch in test_loader:
            a_imgs = batch["arterial"].to(device)
            p_imgs = batch["portal"].to(device)
            labels = batch["time_interval"].float().to(device)

            patient_ids = batch["patient_id"]
            dates = batch["exam_date"]

            outputs = model(a_imgs, p_imgs)

            loss = loss_fn(outputs, labels)
            test_losses.append(loss.item())

            errors = torch.abs(outputs - labels)

            for i in range(len(patient_ids)):
                err = errors[i].item()

                entry = {
                    "patient_id": patient_ids[i],
                    "date": dates[i],
                    "true_interval": labels[i].item(),
                    "pred_interval": outputs[i].item(),
                    "error": err
                }

                # store everything OR only large errors
                if threshold is None or err > threshold:
                    results.append(entry)

                all_errors.append(err)

    avg_test_loss = sum(test_losses) / len(test_losses)
    mae = sum(all_errors) / len(all_errors)

    print(f"Test MSE: {avg_test_loss:.4f}", flush=True)
    print(f"Test MAE: {mae:.4f} seconds", flush=True)

    return results

def main():
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # ------------------------------------------------- Load data -------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------

    path = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/pairs_preprocessed"
    batch_size = 2
    epochs = 20
    
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