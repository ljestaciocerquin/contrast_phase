import pandas as pd
import numpy as np
from pathlib import Path
import os, torch
from tqdm import tqdm
from pathlib import PureWindowsPath
import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")
from Segmentation.cross_segment_models import AttentionFusionSegmentation
from Segmentation.train_cross_seg import SegDataset


from monai.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def get_axial(img, slice_idx):
    # img: [B, C, D, H, W]
    slice_2d = img[0, 0, :, :, slice_idx]

    return slice_2d.squeeze()


def visualize_batch(ap, pvp, gt, preds, i, save_path):
    

    slice_idx = ap.shape[2] // 2

    fig, axs = plt.subplots(1, 4, figsize=(15, 4))

    axs[0].imshow(get_axial(ap, slice_idx).cpu(), cmap="gray")
    axs[0].set_title("AP")

    
    axs[1].imshow(get_axial(pvp, slice_idx).cpu(), cmap="gray")
    axs[1].set_title("PVP")

    axs[2].imshow(get_axial(gt, slice_idx).cpu(), cmap="gray")
    axs[2].set_title("GT")

    axs[3].imshow(get_axial(preds, slice_idx).cpu(), cmap="gray")
    axs[3].set_title("Pred")

    for ax in axs:
        ax.axis("off")

    plt.suptitle(f"Case {i}")
    
    # SAVE instead of only show
    out_file = os.path.join(save_path, f"case_{i:03d}.png")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    plt.savefig(out_file, bbox_inches="tight", dpi=150)

    plt.close()

def main():
    model = AttentionFusionSegmentation(in_channels=1, base=32)
    model.load_state_dict(torch.load("/projects/net_contrast_classification/contrast_phase/Segmentation/trained_Attention.pth", map_location="cpu"))
    model.eval()
    model.cuda()

    data_dir = "/projects/net_contrast_classification/contrast_phase/Preprocessing/Segmentation_data/seg_pairs.csv"
    df = pd.read_csv(data_dir)
    batch_size = 1
    test_dataset = SegDataset(df, split = "test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    save_path = "/projects/net_contrast_classification/contrast_phase/Segmentation/produced_segmentations"

    os.makedirs(save_path, exist_ok=True)


    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            ap = batch["a_img"].cuda()
            pvp = batch["p_img"].cuda()
            gt = batch["lesion"].cuda()

            logits = model(ap, pvp)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.3).float()

            visualize_batch(ap, pvp, gt, preds, i, save_path)

            if i == 10:
                break


if __name__ == "__main__":
    main()