import torch
import os
import pandas as pd
from latent_diffusion import ReconstructionModel, ReconstructionLoss
import torch.nn.functional as F
from monai.data import DataLoader
from monai.transforms import Resize
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import tqdm


class DiffDataset(torch.utils.data.Dataset): 
    """
    CSV-backed paired AP/PVP dataset for the 3D latent diffusion model.
    Expected CSV columns:
        split: train / val / test
        output_path: path to a .pt file saved by paired_preprocess.py

    Each .pt file is expected to contain:
        arterial_image: Tensor shaped [1, D, H, W] or [D, H, W]
        portal_image:   Tensor shaped [1, D, H, W] or [D, H, W]

    Every sample returned here is:
        ap:  [1, 128, 128, 128]
        pvp: [1, 128, 128, 128]    """

    def __init__(
        self,
        csv_path,
        split,
        image_size=(128, 128, 128),
        output_path_col="output_path",
        ap_key="arterial_image",
        pvp_key="portal_image",
        sample=None,
        check_paths=True,
    ):
        self.df = pd.read_csv(csv_path)
        self.split = split
        self.output_path_col = output_path_col
        self.ap_key = ap_key
        self.pvp_key = pvp_key
        self.resize = Resize(spatial_size=image_size, mode="trilinear")

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if sample is not None:
            self.df = self.df.iloc[:sample].reset_index(drop=True)

        if check_paths:
            self._drop_missing_paths()

    def _drop_missing_paths(self):

        exists_mask = self.df[self.output_path_col].apply(os.path.exists)

        self.df = self.df[exists_mask].reset_index(drop=True)

        valid_rows = []

        for idx, row in self.df.iterrows():
            try:
                sample = torch.load(row[self.output_path_col], map_location="cpu")

                ap = sample.get(self.ap_key)
                pvp = sample.get(self.pvp_key)

                ap_liver = sample.get("arterial_liver")
                pvp_liver = sample.get("portal_liver")

                if ap is None or pvp is None:
                    continue
                if ap_liver is None or pvp_liver is None:
                    continue

                valid_rows.append(idx)

            except Exception:
                continue

        self.df = self.df.loc[valid_rows].reset_index(drop=True)
        print(f"{self.split}: {len(self.df)} valid samples")

    def __len__(self):
        return len(self.df)

    def _to_3d_image(self, image):
        image = torch.as_tensor(image, dtype=torch.float32)

        if image.ndim == 3:
            image = image.unsqueeze(0)

        return self.resize(image)

    def concat_masks(self, image, liver_mask):
        return torch.cat([image, liver_mask], dim=0)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        sample = torch.load(row[self.output_path_col], map_location="cpu")

        ap = self._to_3d_image(sample[self.ap_key])
        pvp = self._to_3d_image(sample[self.pvp_key])

        ap_liver = self._to_3d_image(sample["arterial_liver"])
        pvp_liver = self._to_3d_image(sample["portal_liver"])

        ap_liver = (ap_liver > 0).float()
        pvp_liver = (pvp_liver > 0).float()

        ap = self.concat_masks(ap, ap_liver).float()
        pvp = self.concat_masks(pvp, pvp_liver).float()

        return ap, pvp

    @classmethod
    def train_val_test(cls, csv_path, train_split="train", val_split="val", test_split="test", **kwargs):
        return (
            cls(csv_path, split=train_split, **kwargs),
            cls(csv_path, split=val_split, **kwargs),
            cls(csv_path, split=test_split, **kwargs),
        )


# ============================================================
# TRAIN STEP
# ============================================================

def train_step(model, optimizer, criterion, ap, pvp, mode="ap_to_pvp"):

    model.train()
    optimizer.zero_grad()

    reconstructed, predicted_noise, true_noise, z_ap, z_pvp = model(
        ap, pvp, mode=mode
    )

    if mode == "ap_to_pvp":
        target = pvp
        z_source, z_target = z_ap, z_pvp
    else:
        target = ap
        z_source, z_target = z_pvp, z_ap

    diffusion_loss = F.mse_loss(predicted_noise, true_noise) 

    reconstruction_loss = criterion( # the image (reconstructed vs actual) loss + the representation loss (Z_AP vs Z_PVP)
        reconstructed,
        target,
        z_source,
        z_target,
    )

    loss = reconstruction_loss + diffusion_loss

    loss.backward()
    optimizer.step()

    return loss.item(), reconstruction_loss.item(), diffusion_loss.item()


# ============================================================
# VALIDATION STEP
# ============================================================

def validate_step(model, criterion, ap, pvp, mode="ap_to_pvp"):

    model.eval()

    with torch.no_grad():

        reconstructed, predicted_noise, true_noise, z_ap, z_pvp = model(
            ap, pvp, mode=mode
        )

        if mode == "ap_to_pvp":
            target = pvp
            z_source, z_target = z_ap, z_pvp
        else:
            target = ap
            z_source, z_target = z_pvp, z_ap

        diffusion_loss = F.mse_loss(predicted_noise, true_noise)

        reconstruction_loss = criterion(
            reconstructed,
            target,
            z_source,
            z_target,
        )

        loss = reconstruction_loss + diffusion_loss

    return loss.item(), reconstruction_loss.item(), diffusion_loss.item()


# ============================================================
# TRAIN LOOP
# ============================================================

def train_diffusion(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs=10,
    mode="ap_to_pvp",
    early_stopping=None
):

    best_val_loss = float("inf")
    counter = 0
    best_model_state = None

    for epoch in range(epochs):

        train_bar = tqdm(train_loader, desc="Training", leave=False)

        train_loss = train_rec = train_diff = 0.0

        for ap, pvp in train_bar:

            ap, pvp = ap.to(device), pvp.to(device)

            loss, rec_loss, diff_loss = train_step(
                model, optimizer, criterion, ap, pvp, mode
            )

            train_loss += loss
            train_rec += rec_loss
            train_diff += diff_loss

            train_bar.set_postfix(loss=loss, rec_loss=rec_loss, diff_loss=diff_loss)

        train_loss /= len(train_loader)
        train_rec /= len(train_loader)
        train_diff /= len(train_loader)

        # ---------------- VALIDATION SAFE ----------------
        val_loss = val_rec = val_diff = 0.0

        if val_loader is not None:

            val_bar = tqdm(val_loader, desc="Validation", leave=False)

            for ap, pvp in val_bar:

                ap, pvp = ap.to(device), pvp.to(device)

                loss, rec_loss, diff_loss = validate_step(
                    model, criterion, ap, pvp, mode
                )

                val_loss += loss
                val_rec += rec_loss
                val_diff += diff_loss

                val_bar.set_postfix(loss=loss, rec_loss=rec_loss, diff_loss=diff_loss)

            val_loss /= len(val_loader)
            val_rec /= len(val_loader)
            val_diff /= len(val_loader)

        print(
            f"\nEpoch {epoch+1}/{epochs}: "
            f"train_loss={train_loss:.4f} train_rec_loss={train_rec:.4f} train_diff_loss={train_diff:.4f} "
            f"| val_loss={val_loss:.4f} val_rec_loss={val_rec:.4f} val_diff_loss={val_diff:.4f}"
        )

        # ---------------- EARLY STOPPING ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            counter = 0
            print("New best model")
        else:
            counter += 1
            print(f"No improvement ({counter})")

            if early_stopping and counter >= early_stopping:
                print("Early stopping triggered")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


# ============================================================
# EVAL
# ============================================================

def evaluation(model, test_loader, criterion, device, mode="ap_to_pvp"):

    model.eval()

    test_loss = test_rec = test_diff = 0.0

    test_bar = tqdm(test_loader, desc="Testing", leave=False)

    for ap, pvp in test_bar:

        ap, pvp = ap.to(device), pvp.to(device)

        loss, rec_loss, diff_loss = validate_step(
            model, criterion, ap, pvp, mode
        )

        test_loss += loss
        test_rec += rec_loss
        test_diff += diff_loss

    test_loss /= len(test_loader)
    test_rec /= len(test_loader)
    test_diff /= len(test_loader)

    print(
        f"\nTest Loss: {test_loss:.4f} "
        f"Test Reconstruction Loss: {test_rec:.4f} "
        f"Test Diffusion Loss: {test_diff:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = "/projects/net_contrast_classification/contrast_phase/Preprocessing/Paired_diffusion_data/full_pairs.csv"

    batch_size = 1
    epochs = 100
    mode = "ap_to_pvp"

    model = ReconstructionModel(
        latent_dim=512,
        in_channels=2,
        output_size=(128, 128, 128),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    criterion = ReconstructionLoss(lambda_latent=0.1)

    train_dataset, val_dataset, test_dataset = DiffDataset.train_val_test(data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    trained_model = train_diffusion(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        epochs=epochs,
        mode=mode,
    )

    evaluation(trained_model, test_loader, criterion, device, mode)

    save_path = f"Diffusion/trained_LDDM_model_{mode}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(trained_model.state_dict(), save_path)

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()