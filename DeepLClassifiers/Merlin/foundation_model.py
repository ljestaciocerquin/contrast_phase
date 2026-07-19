import torch
from merlin import Merlin
import os, torch
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from monai.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import glob
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")
from DeepLClassifiers.Contrast.contrast_models import evaluate_model, PTDataset
from DeepLClassifiers.Phase_timing.phase_models import class_weights_calculation, train_cnn, OrdinalFocalLoss


class MerlinModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super().__init__()

        self.encoder = Merlin(ImageEmbedding=True)

        feature_dim = 2048  # based on output check

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        z = self.encoder(x)

        # Fix accidental leading singleton dim
        if z.dim() == 3 and z.shape[0] == 1:
            z = z.squeeze(0)

        # Now enforce correct shape
        if z.dim() != 2:
            raise RuntimeError(f"Unexpected shape after fix: {z.shape}")

        return self.head(z)


def main():

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # ------------------------------------------------- Load data -------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------

    data_dir = "/projects/net_contrast_classification/contrast_phase/Preprocessing/Contrast_data/preprocessed_data.csv"

    batch_size = 2
    epochs = 50
    gamma = 4.0
    lambda_ordinal = 0.3

    label_map = {0: "contrast",
                 1: "phase"}
    
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    label_name = label_map[task_id]

    add_augmented = True if label_name == "phase" else False
    train_dataset = PTDataset(data_dir, split = "train", label_name=label_name, encoder=None, add_augmented=add_augmented)
    le = train_dataset.le
    val_dataset = PTDataset(data_dir, split = "val", label_name=label_name, encoder=le)
    test_dataset = PTDataset(data_dir, split = "test", label_name=label_name, encoder=le)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for batch in train_loader:
        images = batch["image"]
        labels = batch["label"]
        print("Image shape:", images.shape, "Labels:", labels, flush=True)
        break

    # --------------------------------------------------- Train ---------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------
    

    model = MerlinModel(num_classes=len(le.classes_), dropout_rate=0.3)

    print(f"\nTraining for task:{label_name} using Merlin embedding\n")

    save_path = f"/projects/net_contrast_classification/contrast_phase/DeepLClassifiers/Merlin/{label_name}"
    os.makedirs(save_path, exist_ok=True)

    class_weights = class_weights_calculation(data_dir) if label_name == "phase" else None
    print("Class weights:", class_weights, flush=True)

    loss_fn = (
        OrdinalFocalLoss(alpha=class_weights, gamma=gamma, lambda_ordinal=lambda_ordinal, reduction="mean") 
        if label_name == "phase" else CrossEntropyLoss()
    )

    trained_model = train_cnn(model,
                                    train_loader,
                                    class_weights=class_weights,
                                    val_loader=val_loader,
                                    epochs = epochs,
                                    early_stopping=10,
                                    loss_fn = loss_fn,
                                    save_path = save_path)
    

    # Saving the trained model
    save_path_model = (
                    f"{save_path}/trained_models/"
                    f"{label_name}{f'_g{int(gamma)}' if label_name == 'phase' else ''}_trained.pth"
                )

    try:
        # ensure directory exists
        os.makedirs(os.path.dirname(save_path_model), exist_ok=True)

        torch.save(trained_model.state_dict(), save_path_model)
        print(f"Model saved successfully at: {save_path_model}", flush=True)

    except Exception as e:
        print(f"Error saving model: {e}", flush=True)

    # ------------------------------------------------- Evaluate --------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------
        
    _, misclassifications = evaluate_model(trained_model, test_loader, class_names=le.classes_)

    df = pd.DataFrame(misclassifications)
    df.to_csv(f"{save_path}/results/{label_name}{[f'_g{int(gamma)}' if label_name == 'phase' else '']}_misclassifications.csv", index=False)


if __name__ == "__main__":
    main()