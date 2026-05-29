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
import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")
from DeepLClassifiers.Contrast.contrast_models import CNN8, ResNet

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

class CNN8_encoder(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, 3, padding=1),
                nn.InstanceNorm3d(out_c, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, 3, padding=1),
                nn.InstanceNorm3d(out_c, affine=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.MaxPool3d(2)
            )

        self.features = nn.Sequential(
            block(1, 16),
            block(16, 32),
            block(32, 64),
            block(64, 128),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)   # [B, 128]
        return x
    
class DeepTimeEstimator(nn.Module):
    def __init__(self, encoder, embedding_dim, t_min=0.0, t_max=60.0):
        super().__init__()

        self.encoder = encoder
        self.time_head = nn.Linear(embedding_dim, 1)

        self.t_min = t_min
        self.t_max = t_max

    def encode(self, x):
        return self.encoder(x)   # nothing else

    def forward(self, a_img, p_img):

        z_a = self.encode(a_img)
        z_p = self.encode(p_img)

        z_a = F.normalize(z_a, dim=1)
        z_p = F.normalize(z_p, dim=1)

        s_a = self.time_head(z_a)
        s_p = self.time_head(z_p)

        delta = s_p - s_a           # this can be (-inf, +inf)
        out = torch.sigmoid(delta)  # sigmoid maps the difference to (0,1) - this is the temporal progression score

        time = out * (self.t_max - self.t_min) + self.t_min      # the sigmoid score is mapped back to real time

        return time.squeeze(1)
    


    def forward(self, a_img, p_img):
        # initial shape of inputs = [B, 1, D, H, W]
        
        # encode both phases => Siamese modelling 
        z_a = self.encode(a_img)            # [B, 512]
        z_p = self.encode(p_img)            # [B, 512]

        z_a = F.normalize(z_a, dim=1)
        z_p = F.normalize(z_p, dim=1)

        s_a = self.time_head(z_a)   # [B, 1]
        s_p = self.time_head(z_p)   # [B, 1]

        delta = s_p - s_a           # temporal difference
        out = torch.sigmoid(delta)  # [0,1]

        # convert back to time
        time = out * (self.t_max - self.t_min) + self.t_min

        return time.squeeze(1)
    
class DeepTimeRegressor(nn.Module):
    def __init__(self, encoder, embedding_dim=512):
        super().__init__()

        # Encoder 
        self.encoder = encoder

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim * 3 + 1, 1024),  # add layers to slowly reduce dimensionallity 
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )


    def encode(self, x):
        x = self.encoder(x)              # already [B, 512]
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        return x

    def forward(self, a_img, p_img):
        # initial shape of inputs = [B, 1, D, H, W]
        
        # encode both phases => Siamese modelling 
        z_a = self.encode(a_img)            # [B, 512]
        z_p = self.encode(p_img)            # [B, 512]

        z_a = F.normalize(z_a, dim=1)
        z_p = F.normalize(z_p, dim=1)

        # feature interactions
        z_diff = z_p - z_a                                           # captures the changing contrast dynamics from arterial -> portal
        z_cos = F.cosine_similarity(z_a, z_p, dim=1).unsqueeze(1)   # captures the similarity between arterial and portal respresentations

        # concatenate
        z = torch.cat([z_a, z_p, z_diff, z_cos], dim=1)

        # predict time interval
        out = self.regressor(z)

        return out.squeeze(1)

# class RadiomicsTimeEstimator():


# class SeqTimeEstimator():
#     def __init__(self, feature_dim =128, hidden_size=128, num_layers=1):
#         super().__init__()

#         self.feature_dim = feature_dim

#         # Encoder 
#         self.cnn = Small3DCNN_v2(num_classes=feature_dim)

#         # --- LSTM ---
#         self.lstm = nn.LSTM(
#             input_size=feature_dim,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True
#         )

#         # --- Regression head ---
#         self.fc = nn.Linear(hidden_size, 1)


#     def forward(self, x):
#         # x: (batch size, time points (arterial and portal = 2), channels, D, H, W)
#         B, T, C, D, H, W = x.shape

#         # merge batch and time
#         x = x.view(B * T, C, D, H, W)

#         # CNN features
#         features = self.cnn(x)  # (B*T, feature_dim)

#         # reshape back
#         features = features.view(B, T, -1)

#         # LSTM
#         lstm_out, _ = self.lstm(features)

#         # last timestep
#         out = lstm_out[:, -1, :]

#         # regression
#         out = self.fc(out)

#         return out.squeeze(1)




def train_regressor(model, 
                    encoder_name,
                    train_loader, 
                    val_loader = None, 
                    epochs=10,
                    lr=1e-4, 
                    weight_decay = 1e-4, 
                    early_stopping: Optional[int] = None,
                    error_weights = 0.3
                    ):
    

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

            if error_weights:
                weights = torch.ones_like(labels)
                weights[(labels >= 30) & (labels <= 40)] = error_weights
                loss = (weights * error).mean()
            else:
                loss = error.mean()

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

        if train_curve is not None and val_curve is not None:
            save_path = "/projects/net_contrast_classification/contrast_phase/DeepLClassifiers/Time_interval/results"
            os.makedirs(save_path, exist_ok=True)

            plt.clf()  
            plt.plot(train_curve, label="train MAE")
            plt.plot(val_curve, label="val MAE")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Time Interval Estimation with {model.encoder.__class__.__name__}")
            plt.legend()
            plt.pause(0.01)
            plt.savefig(f"{save_path}/{model.__class__.__name__}_{encoder_name}_training_MAE_curve.png") 


    # Loading the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

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


def build_encoder(name):
    if name == "ResNet10":
        enc = resnet10(
            spatial_dims=3,
            n_input_channels=1,
            pretrained=False,
        )
        enc.fc = nn.Identity()
        return enc, 512

    elif name == "CNN8":
        return CNN8_encoder(), 128

    else:
        raise ValueError(f"Unknown encoder: {name}")
    
def build_model(model_name):

    encoder_name, task = model_name.split("_")

    encoder, emb_dim = build_encoder(encoder_name)

    if task == "estim":
        model = DeepTimeEstimator(encoder, emb_dim)

    elif task == "reg":
        model = DeepTimeRegressor(encoder, emb_dim)

    else:
        raise ValueError(f"Unknown task: {task}")

    return model, encoder_name


def main():
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # ------------------------------------------------- Load data -------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------

    data_dir = "/projects/net_contrast_classification/contrast_phase/Preprocessing/Time_interval_data/paired_preprocessed_data.csv"

    batch_size = 1
    epochs = 100
    
    model_map = {0: "ResNet10_estim", 1: "CNN8_estim",
                 2: "ResNet10_reg", 3: "CNN8_reg"
                # , 2: "Merlin"
                }
    
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    model_name = model_map[task_id]


    train_dataset = PairDataset(data_dir, split = "train", filter_outliers=True)
    val_dataset = PairDataset(data_dir, split = "val", filter_outliers=True)
    test_dataset = PairDataset(data_dir, split = "test", filter_outliers=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # --------------------------------------------------- Train ---------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------
    

    model, encoder_name = build_model(model_name)


    print(
    f"Training time interval estimation model using "
    f"{model.__class__.__name__} with {encoder_name} encoder",
    flush=True
    )   

    trained_model = train_regressor(model, encoder_name,
                                        train_loader,
                                        val_loader,
                                        epochs = epochs,
                                        early_stopping=None,
                                        error_weights=None,
                                        )

    save_path = "/projects/net_contrast_classification/contrast_phase/DeepLClassifiers/Time_interval"
    os.makedirs(f"{save_path}/trained_models", exist_ok=True)
    os.makedirs(f"{save_path}/results", exist_ok=True)

    # Saving the trained model
    save_path_model = f"{save_path}/trained_models/{model.__class__.__name__}_{encoder_name}_trained_model.pth"

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
    df.to_csv(f"{save_path}/results/{model.__class__.__name__}_{encoder_name}_pred_intervals.csv", index=False)   

if __name__ == "__main__":
    main()
