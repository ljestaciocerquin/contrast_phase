import torch
import os
import pandas as pd

from latent_diffusion_v3 import (
    ReconstructionModel,
    ReconstructionLoss,
    AutoEncoderModel,
    AutoencoderLoss,
)
from monai.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch.nn.functional as F

# ============================================================
# DATASET
# ============================================================

class DiffDataset(torch.utils.data.Dataset):
    """
    CSV-backed paired AP/PVP dataset for the 3D latent diffusion model.

    Expected CSV columns:
        split: train / val / test
        output_path: path to a .pt file saved by paired_preprocess.py

    Each .pt file is expected to contain:
        arterial_image: Tensor shaped [1, D, H, W] or [D, H, W]
        portal_image:   Tensor shaped [1, D, H, W] or [D, H, W]
        arterial_liver: Tensor shaped [1, D, H, W] or [D, H, W]
        portal_liver:   Tensor shaped [1, D, H, W] or [D, H, W]

    Every sample returned here is:
        ap:  [2, 128, 128, 128]
        pvp: [2, 128, 128, 128]

    Channel 0 = image
    Channel 1 = liver mask
    """

    def __init__(
        self,
        csv_path,
        split,
        # image_size=(128, 128, 128),
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


        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if sample is not None:
            self.df = self.df.iloc[:sample].reset_index(drop=True)

        if check_paths:
            self._drop_missing_paths()

    def _drop_missing_paths(self):

        exists_mask = self.df[
            self.output_path_col
        ].apply(os.path.exists)

        self.df = self.df[exists_mask].reset_index(drop=True)

        valid_rows = []

        for idx, row in self.df.iterrows():

            try:

                sample = torch.load(row[self.output_path_col],map_location="cpu")

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

        if image.shape != (1, 128, 128, 128):
            raise ValueError(
                f"Expected shape [1,128,128,128], got {image.shape}"
            )

        return image

    def concat_masks(self,image,liver_mask):

        return torch.cat(
            [image,liver_mask],
            dim=0,
        )

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        sample = torch.load(
            row[self.output_path_col],
            map_location="cpu",
        )

        ap = self._to_3d_image(sample[self.ap_key])
        pvp = self._to_3d_image(sample[self.pvp_key])

        ap_liver = self._to_3d_image(sample["arterial_liver"])
        pvp_liver = self._to_3d_image(sample["portal_liver"])

        ap_liver = (ap_liver > 0).float()
        pvp_liver = (pvp_liver > 0).float()

        ap = self.concat_masks(ap,ap_liver).float()
        pvp = self.concat_masks(pvp,pvp_liver).float()

        return ap, pvp

    @classmethod
    def train_val_test(
        cls,
        csv_path,
        train_split="train",
        val_split="val",
        test_split="test",
        **kwargs,
    ):

        return (
            cls(csv_path, split=train_split, **kwargs),
            cls(csv_path, split=val_split, **kwargs),
            cls(csv_path, split=test_split, **kwargs),
        )

# ============================================================
# AUTOENCODER TRAINING
# ============================================================

def train_autoencoder(model, loader, optimizer, criterion, device, epochs=10):

    model.to(device)

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        recon_stats = []

        bar = tqdm(loader, desc=f"[AE] Epoch {epoch+1}")

        for ap, pvp in bar:

            ap = ap.to(device, non_blocking=True)
            pvp = pvp.to(device, non_blocking=True)

            # =========================
            # AP STEP
            # =========================
            optimizer.zero_grad(set_to_none=True)

            recon_ap, _ = model(ap)
            loss_ap = criterion(recon_ap, ap)

            loss_ap.backward()
            optimizer.step()

            # =========================
            # PVP STEP
            # =========================
            optimizer.zero_grad(set_to_none=True)

            recon_pvp, _ = model(pvp)
            loss_pvp = criterion(recon_pvp, pvp)

            loss_pvp.backward()
            optimizer.step()

            # =========================
            # LOGGING
            # =========================
            loss = 0.5 * (loss_ap.item() + loss_pvp.item())

            total_loss += loss

            recon_stats.append(recon_ap.detach().abs().mean().item())
            recon_stats.append(recon_pvp.detach().abs().mean().item())

            bar.set_postfix(loss=loss)

            # IMPORTANT FOR 3D MEMORY
            del recon_ap, recon_pvp
            del loss_ap, loss_pvp
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(loader)
        avg_recon = sum(recon_stats) / len(recon_stats)

        print(f"\n[AE] Epoch {epoch+1}")
        print(f"  Loss: {avg_loss:.6f}")
        print(f"  Recon |mean|: {avg_recon:.6f}")

        if avg_recon < 1e-5:
            print("WARNING: AE COLLAPSE (near-zero outputs)")

        if avg_loss > 5:
            print("WARNING: AE instability (loss too high)")

    return model


# ============================================================
# DIFFUSION TRAINING
# ============================================================

def train_diffusion_step(model, optimizer, ap, pvp, criterion, mode):

    model.train()
    optimizer.zero_grad()

    pred_noise, true_noise, z_pred, z_tgt, recon = model(ap, pvp, mode=mode)

    # diffusion loss
    loss_diff = criterion(pred_noise, true_noise)

    # IMPORTANT FIX: per-sample normalization (NOT global)
    z_pred = (z_pred - z_pred.mean(dim=(1,2,3,4), keepdim=True)) / (
        z_pred.std(dim=(1,2,3,4), keepdim=True) + 1e-6
    )
    z_tgt = (z_tgt - z_tgt.mean(dim=(1,2,3,4), keepdim=True)) / (
        z_tgt.std(dim=(1,2,3,4), keepdim=True) + 1e-6
    )

    loss_latent = F.mse_loss(z_pred, z_tgt)

    target = pvp if mode == "ap_to_pvp" else ap
    loss_img = F.l1_loss(recon, target)

    loss = loss_diff + 0.1 * loss_latent + 0.01 * loss_img

    loss.backward()
    optimizer.step()

    return loss.item(), loss_diff.item(), loss_latent.item(), loss_img.item()

def validate_diffusion(model, ap, pvp, criterion, mode):

    model.eval()

    with torch.no_grad():

        pred_noise, true_noise, z_pred, z_tgt, recon = model(ap, pvp, mode=mode)

        loss_diff = criterion(pred_noise, true_noise)

        z_pred = (z_pred - z_pred.mean(dim=(1,2,3,4), keepdim=True)) / (
            z_pred.std(dim=(1,2,3,4), keepdim=True) + 1e-6
        )
        z_tgt = (z_tgt - z_tgt.mean(dim=(1,2,3,4), keepdim=True)) / (
            z_tgt.std(dim=(1,2,3,4), keepdim=True) + 1e-6
        )

        loss_latent = F.mse_loss(z_pred, z_tgt)

        target = pvp if mode == "ap_to_pvp" else ap
        loss_img = F.l1_loss(recon, target)

        loss = loss_diff + 0.1 * loss_latent + 0.01 * loss_img

    return loss.item(), loss_diff.item(), loss_latent.item(), loss_img.item()

def train_diffusion(model, train_loader, val_loader,
                    optimizer, criterion, device, mode = "ap_to_pvp" ,epochs=50):

    best = float("inf")

    for epoch in range(epochs):

        train_loss = 0
        train_loss_diff = 0
        train_loss_latent = 0
        train_loss_img = 0

        bar = tqdm(train_loader, desc=f"Diff Epoch {epoch+1}")

        for ap, pvp in bar:

            ap = ap.to(device)
            pvp = pvp.to(device)

            loss, loss_diff, loss_latent, loss_img = train_diffusion_step(model, optimizer, ap, pvp, criterion, mode)
            train_loss += loss
            train_loss_diff += loss_diff
            train_loss_latent += loss_latent
            train_loss_img += loss_img


            bar.set_postfix(loss=loss)

        train_loss /= len(train_loader)
        train_loss_diff /= len(train_loader)
        train_loss_latent /= len(train_loader)
        train_loss_img /= len(train_loader)


        val_loss = 0
        val_loss_diff = 0
        val_loss_latent = 0
        val_loss_img = 0

        if val_loader is not None:

            for ap, pvp in val_loader:
                ap, pvp = ap.to(device), pvp.to(device)
                loss, loss_diff, loss_latent, loss_img = validate_diffusion(model, ap, pvp, criterion, mode)
                val_loss += loss
                val_loss_diff += loss_diff
                val_loss_latent += loss_latent
                val_loss_img += loss_img

            val_loss /= len(val_loader)
            val_loss_diff /= len(val_loader)
            val_loss_latent /= len(val_loader)
            val_loss_img /= len(val_loader)


        print(
            f"""
        Epoch {epoch+1:03d}
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        TRAIN
        Total Loss     : {train_loss:.4f}
        Diffusion Loss : {train_loss_diff:.4f}
        Latent Loss    : {train_loss_latent:.4f}
        Image Loss     : {train_loss_img:.4f}

        VALIDATION
        Total Loss     : {val_loss:.4f}
        Diffusion Loss : {val_loss_diff:.4f}
        Latent Loss    : {val_loss_latent:.4f}
        Image Loss     : {val_loss_img:.4f}
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        )

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "Diffusion/best_diffusion.pth")
            print("Saved best model")

    return model

def evaluate_diffusion(model, test_loader, criterion, device, mode="ap_to_pvp"):

    model.eval()
    test_loss = 0.0
    test_loss_diff = 0.0
    test_loss_latent = 0.0
    test_loss_img = 0.0



    with torch.no_grad():
        for ap, pvp in test_loader:
            ap, pvp = ap.to(device), pvp.to(device)
            pred_noise, true_noise, z_pred, z_tgt, recon = model(ap, pvp, mode=mode)

            # 1. diffusion loss (ALWAYS required)
            loss_diff = criterion(pred_noise, true_noise)

            # 2. latent alignment loss (NEW, IMPORTANT)
            # Normalization
            z_pred = (z_pred - z_pred.mean(dim=(1,2,3,4), keepdim=True)) / (z_pred.std(dim=(1,2,3,4), keepdim=True) + 1e-6)
            z_tgt  = (z_tgt  - z_tgt.mean(dim=(1,2,3,4), keepdim=True)) / (z_tgt.std(dim=(1,2,3,4), keepdim=True) + 1e-6)

            loss_latent = F.mse_loss(z_pred, z_tgt, reduction="mean")

            # 3. image loss (ONLY if decoder output is meaningful)
            loss_img = F.l1_loss(recon, pvp, reduction="mean") if mode == "ap_to_pvp" else F.l1_loss(recon, ap, reduction="mean")

            # combine the losses
            loss = loss_diff + 0.1 * loss_latent + 0.01 * loss_img
            
            test_loss += loss.item()
            test_loss_diff += loss_diff.item()
            test_loss_latent += loss_latent.item()
            test_loss_img += loss_img.item()



    test_loss /= len(test_loader)
    test_loss_diff /= len(test_loader)
    test_loss_latent /= len(test_loader)
    test_loss_img /= len(test_loader)


    print(
        f"""
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    TEST RESULTS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Total Loss : {test_loss:.4f}
    Diffusion Loss: {test_loss_diff:.4f}
    Latent Loss   : {test_loss_latent:.4f}
    Image Loss    : {test_loss_img:.4f}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    )

    return test_loss


# ============================================================
# MAIN
# ============================================================

def main():
    print("TRAINING VERSION 3")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = (
        "/projects/net_contrast_classification/"
        "contrast_phase/Preprocessing/"
        "Paired_diffusion_data/full_pairs.csv"
    )

    batch_size = 1
    mode = "ap_to_pvp"

    # ========================================================
    # DATA
    # ========================================================

    train_dataset, val_dataset, test_dataset = DiffDataset.train_val_test(data_dir)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=2)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=2)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=2)

    # ========================================================
    # STAGE 1: AUTOENCODER
    # ========================================================
    ae_path = "/projects/net_contrast_classification/contrast_phase/Diffusion/autoencoder.pth"

    ae = AutoEncoderModel(in_channels=2).to(device)

    if os.path.exists(ae_path):

        print("\n=== Loading pretrained autoencoder ===")
        ae.load_state_dict(torch.load(ae_path, map_location=device))

    else:
        print("\n=== Stage 1: Autoencoder training ===")

        ae = AutoEncoderModel(in_channels=2).to(device)

        ae_loss = AutoencoderLoss(lambda_mask=0.1)
        ae_opt = torch.optim.Adam(ae.parameters(), lr=1e-4)

        ae = train_autoencoder(
            ae, train_loader, ae_opt, ae_loss, device, epochs=20
        )

        torch.save(ae.state_dict(), "Diffusion/autoencoder.pth")

    # ========================================================
    # STAGE 2: DIFFUSION
    # ========================================================

    print("\n=== Stage 2: Latent Diffusion training ===")

    diffusion_model = ReconstructionModel(in_channels=2).to(device)

    diffusion_model.load_pretrained_autoencoder(ae)
    diffusion_model.freeze_autoencoder()

    diffusion_opt = torch.optim.Adam(diffusion_model.diffusion.parameters(), lr=1e-4)

    diffusion_loss = ReconstructionLoss()

    diffusion_model = train_diffusion(
        diffusion_model,
        train_loader,
        val_loader,
        diffusion_opt,
        diffusion_loss,
        device,
        mode = mode, 
        epochs=200
    )

    torch.save(diffusion_model.state_dict(), "Diffusion/diffusion.pth")

    evaluate_diffusion(
        diffusion_model,
        test_loader,
        diffusion_loss,
        device,
        mode=mode
    )


if __name__ == "__main__":
    main()