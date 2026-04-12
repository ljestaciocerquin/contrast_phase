import os, random, torch
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from monai.data import Dataset, DataLoader
from monai.networks.nets import resnet18
from torch.nn import CrossEntropyLoss
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# from monai.data import MetaTensor
# from sklearn.utils.class_weight import compute_class_weight
# from torch.utils.data import WeightedRandomSampler


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class PTDataset(Dataset):
    def __init__(self, folder, label_name="contrast", encoder=None, sample=None):
        self.files = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".pt")
        ])

        if sample:
            self.files = self.files[:sample]

        self.label_name = label_name

        # -------- Load labels --------
        self.raw_labels = []
        self.keys = []

        for f in self.files:
            data = torch.load(f, map_location="cpu") 
            self.raw_labels.append(str(data[label_name]))
            self.keys.append(data["key"])           

        # -------- Encode labels --------
        if encoder is None:
            self.le = LabelEncoder()
            self.le.fit(self.raw_labels)
        else:
            self.le = encoder

        self.encoded_labels = self.le.transform(self.raw_labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")

        image = data["image"]


        return {
            "key": self.keys[idx],                          
            "image": image,
            "label": torch.tensor(self.encoded_labels[idx], dtype=torch.long)
            }
    
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

def train_cnn(model, train_loader, weights=None, val_loader=None, epochs=10, lr=1e-4, weight_decay = 1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if weights is not None:
        weights = weights.to(device)

    loss_fn = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()


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

                if counter >= patience:
                    print("Early stopping triggered")
                    break

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
    misclassifications = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            keys = batch["key"]

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            for i in range(len(keys)):
                if preds[i] != labels[i]:
                    misclassifications.append({
                        "key": keys[i],
                        "true": class_names[labels[i].item()],
                        "pred": class_names[preds[i].item()]
                    })

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

    return metrics, misclassifications

class Small3DCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.layers = nn.Sequential(
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

        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class Small3DCNN_1(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        # AdaptiveAvgPool3d(1) adaptively collapses spatial dims
        self.adaptive_pool = nn.AdaptiveAvgPool3d(1)

        # Classifier
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # flatten the feature map
        x = self.adaptive_pool(x)  # shape: (batch, 32, 1, 1, 1)
        x = torch.flatten(x, 1)    # shape: (batch, 32)

        out = self.classifier(x)
        return out
    


def main():

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # ------------------------------------------------- Load data -------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------

    path = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/contrast_preprocessed"
    batch_size = 5
    epochs = 10

    label_map = {0: "contrast", 1: "phase"}
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    label_name = label_map[task_id]

    print(f"Training 3D CNN for contrast phase classification with label: {label_name}", flush=True)

    train_dataset = PTDataset(f"{path}/train", label_name=label_name, encoder=None)
    le = train_dataset.le
    val_dataset = PTDataset(f"{path}/val", label_name=label_name, encoder=le)
    test_dataset = PTDataset(f"{path}/test", label_name=label_name, encoder=le)

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
    print("Model: Small3DCNN", flush=True)

    # Simple 3D CNN trained from scratch
    model = Small3DCNN(num_classes=len(le.classes_))


    # model = resnet18(
    #         spatial_dims=3,
    #         n_input_channels=1,   
    #         num_classes=len(le.classes_),
    #         pretrained = False
    # )

    trained_model = train_cnn(model,
                            train_loader,
                            weights = None,
                            val_loader=val_loader,
                            epochs = epochs)
    

    # ------------------------------------------------- Evaluate --------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------
        
    # Saving the trained model
    save_path = f"/projects/net_contrast_classification/contrast_phase/DeepLClassifiers/{label_name}_trained_model.pth"

    try:
        # ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(trained_model.state_dict(), save_path)
        print(f"Model saved successfully at: {save_path}", flush=True)

    except Exception as e:
        print(f"Error saving model: {e}", flush=True)

    
    # Evaluation 
    _, misclassifications = evaluate_model(trained_model, test_loader, class_names=le.classes_)

    df = pd.DataFrame(misclassifications)
    df.to_csv(f"/projects/net_contrast_classification/contrast_phase/DeepLClassifiers/{label_name}_misclassifications.csv", index=False)


if __name__ == "__main__":
    main()