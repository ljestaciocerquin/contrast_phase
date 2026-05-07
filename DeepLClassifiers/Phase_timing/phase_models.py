from pyexpat import model
import os, random, torch
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from monai.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings

import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")
from DeepLClassifiers.Contrast.contrast_models import PTDataset, evaluate_model, CNN8, ResNet
warnings.filterwarnings("ignore", category=FutureWarning)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

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

        pt = torch.exp(-ce_loss)                # probability of correct class

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

def class_weights_calculation(data_dir, split = "train", label_name="phase"):
    data = pd.read_csv(data_dir)
    train_data = data[data["split"] == split]
    label_counts = train_data[label_name].value_counts().sort_index()
    total_samples = len(train_data)
    num_classes = len(label_counts)

    class_weights = total_samples / (num_classes * label_counts)
    class_weights = class_weights ** 0.5                                            # square root smoothing to prevent extreme weights
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32)

    return class_weights_tensor

def train_cnn(model,
              train_loader, 
              class_weights=None, 
              val_loader=None, 
              epochs=10, 
              lr=1e-4, 
              weight_decay = 1e-4, 
              early_stopping = None,
              save_path = None
              ):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)


    loss_fn = FocalLoss(alpha=class_weights, gamma=2.0) if class_weights is not None else CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    counter = 0
    best_model_state = None

    train_loss_curve = []
    val_loss_curve = []

    train_acc_curve = []
    val_acc_curve = []

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

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()


            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)                # outputs = logits => pick the highest logit as the class label
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            loop.set_postfix(
                loss=loss.item(),
                acc=train_correct / train_total if train_total else 0.0
            )

        avg_train_loss = train_loss / len(train_loader)
        train_loss_curve.append(avg_train_loss)
        train_acc = train_correct / train_total if train_total else 0.0
        train_acc_curve.append(train_acc)


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


                    outputs = model(images)
                    loss = loss_fn(outputs, labels)

                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_loss_curve.append(avg_val_loss)
            val_acc = val_correct / val_total if val_total else 0.0
            val_acc_curve.append(val_acc)

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
                print(f"No improvement ({counter})")

                if early_stopping is not None and counter >= early_stopping:
                    print("Early stopping triggered")
                    break
                    
        if save_path is not None:
            results_dir = f"{save_path}/results"
            os.makedirs(results_dir, exist_ok=True)
            loss_str = f", FocalLoss (gamma={loss_fn.gamma})" if isinstance(loss_fn, FocalLoss) else ""

            # ---- Loss curve ----
            if train_loss_curve is not None and val_loss_curve is not None:
                plt.figure()
                plt.plot(train_loss_curve, label="train loss")
                plt.plot(val_loss_curve, label="val loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Loss curve: {model.__class__.__name__}{loss_str}")
                plt.legend()

                plt.savefig(f"{results_dir}/{model.__class__.__name__}_loss_curve.png")
                plt.close()

            # ---- Accuracy curve ----
            if train_acc_curve is not None and val_acc_curve is not None:
                plt.figure()
                plt.plot(train_acc_curve, label="train acc")
                plt.plot(val_acc_curve, label="val acc")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title(f"Accuracy curve: {model.__class__.__name__}{loss_str}")
                plt.legend()

                plt.savefig(f"{results_dir}/{model.__class__.__name__}_acc_curve.png")
                plt.close()

    # Loading the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)


    return model

def main():

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # ------------------------------------------------- Load data -------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------

    data_dir = "/projects/net_contrast_classification/contrast_phase/Preprocessing/Contrast_data/preprocessed_data.csv"

    batch_size = 2
    epochs = 50

    model_map = {0: "ResNet10", 1: "CNN8"}
    
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    model_name = model_map[task_id]


    train_dataset = PTDataset(data_dir, split = "train", label_name="phase", encoder=None, add_augmented=True)
    le = train_dataset.le
    val_dataset = PTDataset(data_dir, split = "val", label_name="phase", encoder=le)
    test_dataset = PTDataset(data_dir, split = "test", label_name="phase", encoder=le)


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
    

    if model_name == "ResNet10":
        model = ResNet(num_classes = len(le.classes_), dropout_rate=0.3)
    
    elif model_name == "CNN8":
        # Simple 3D CNN trained from scratch
        model = CNN8(num_classes=len(le.classes_), dropout_rate=0.2)

    # elif model_name == "ViT":
    #     model = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True, dropout_rate = 0.3, save_attn = True)
    
    # else:
    #     model = MerlinModel(num_classes=len(le.classes_), dropout_rate=0.2)


    print(f"\nTraining {model_name} for phase timing classification\n")

    save_path = "/projects/net_contrast_classification/contrast_phase/DeepLClassifiers/Phase_timing"
    os.makedirs(save_path, exist_ok=True)

    class_weights = class_weights_calculation(data_dir)
    print("Class weights:", class_weights, flush=True)

    trained_model = train_cnn(model,
                                    train_loader,
                                    class_weights=class_weights,
                                    val_loader=val_loader,
                                    epochs = epochs,
                                    early_stopping=10,
                                    save_path = save_path)
    

    # Saving the trained model
    save_path_model = f"{save_path}/trained_models/{model_name}_trained.pth"

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
    df.to_csv(f"{save_path}/results/{model_name}_misclassifications.csv", index=False)


if __name__ == "__main__":
    main()