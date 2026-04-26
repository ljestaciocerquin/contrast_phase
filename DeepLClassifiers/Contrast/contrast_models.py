import os, random, torch
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from monai.data import Dataset, DataLoader
from monai.networks.nets import resnet18, resnet10
# from merlin import Merlin
from torch.nn import CrossEntropyLoss
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class PTDataset(Dataset):
    def __init__(self,
                 csv_path,
                 split,
                 label_name="contrast",
                 encoder=None,
                 sample=None,
                 add_augmented=False):

        self.df = pd.read_csv(csv_path)
        self.split = split
        self.label_name = label_name

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if sample:
            self.df = self.df.iloc[:sample].reset_index(drop=True)

        self._check_paths()

        original_len = len(self.df)


        # ---------------- label encoding (GLOBAL) ----------------

        if encoder is None:
            self.le = LabelEncoder()
            self.le.fit(self.df[self.label_name].astype(str))
        else:
            self.le = encoder

        self.df["encoded_label"] = self.le.transform(
            self.df[self.label_name].astype(str)
        )

        # ---------------- optional inclusion of augmented images ----------------
        if add_augmented and self.split == "train":
            self._include_augmentations()
            self._check_paths()
            print(f"[INFO] Dataset expanded: {original_len} -> {len(self.df)}")
        else:
            print(f"[INFO] {str.capitalize(split)} dataset size: {original_len}")



    def _check_paths(self):
        mask = self.df["output_path"].apply(os.path.exists)

        missing = self.df[~mask]
        if len(missing) > 0:
            print(f"[WARNING] Missing files: {len(missing)}")
            print(missing["output_path"].head().tolist())

        self.df = self.df[mask].reset_index(drop=True)


    def _include_augmentations(self):                       # ADDS ONLY RARE CLASS AUGMENTED IMAGES AND IGNORES THE AUGMENTATIONS OF THE COMMON CLASS IMAGES
        new_rows = []

        for _, row in self.df.iterrows():
            base_path = row["output_path"]

            # always include original
            new_rows.append(row.copy())

            # only for train split and augment == 1
            if row.get("augment", 0) == 1:
                base_no_ext = os.path.splitext(base_path)[0]

                aug_paths = glob.glob(base_no_ext + "_aug*.pt")
                if len(aug_paths) == 0:
                    print(f"[WARNING] No augmentations found for {base_path}")

                for aug_path in aug_paths:
                    if os.path.exists(aug_path):  # optional (glob already ensures this)
                        new_row = row.copy()
                        new_row["output_path"] = aug_path
                        new_rows.append(new_row)

        self.df = pd.DataFrame(new_rows).reset_index(drop=True)
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        data = torch.load(row["output_path"], map_location="cpu", weights_only=False)

        return {
            "key": row["MatchKey"],
            "image": torch.as_tensor(data["image"], dtype=torch.float32),
            "label": torch.tensor(row["encoded_label"], dtype=torch.long),
        }
          
def train_cnn(model,
              train_loader, 
              weights=None, 
              val_loader=None, 
              epochs=10, 
              lr=1e-4, 
              weight_decay = 1e-4, 
              early_stopping = None,
              save_path = None
              ):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if weights is not None:
        weights = weights.to(device)


    loss_fn = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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

            # Plot loss curve
            if train_loss_curve is not None and val_loss_curve is not None:
                plt.clf()
                plt.plot(train_loss_curve, label="train loss")
                plt.plot(val_loss_curve, label="val loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Contrast loss curve: {model.__class__.__name__}")
                plt.legend()
                plt.pause(0.01)
                plt.savefig(f"{results_dir}/{model.__class__.__name__}_loss_curve.png") 

            # Plot accuracy curve
            if train_acc_curve is not None and val_acc_curve is not None:
                plt.clf()
                plt.plot(train_acc_curve, label="train acc")
                plt.plot(val_acc_curve, label="val acc")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title(f"Contrast accuracy curve: {model.__class__.__name__}")
                plt.legend()
                plt.pause(0.01)
                plt.savefig(f"{results_dir}/{model.__class__.__name__}_acc_curve.png") 

    # Loading the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)


    return model


def evaluate_model(model,test_loader, device=None, class_names=None):
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

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class CNN8(nn.Module):
    def __init__(self, num_classes=6, dropout_rate = 0.2):
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
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class ResNet(nn.Module):
    def __init__(self, num_classes, dropout_rate):
        super().__init__()

        self.net = resnet10(
                            spatial_dims=3,
                            n_input_channels=1,   
                            num_classes=num_classes,
                            pretrained = False,
                            )

        # replace classifier
        in_features = self.net.fc.in_features

        self.net.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# class MerlinModel(nn.Module):
#     def __init__(self, num_classes, dropout_rate=0.3):
#         super().__init__()

#         # load pretrained Merlin encoder
#         self.encoder = Merlin(ImageEmbedding=True)

#         #  find embedding size (you may need to adjust this!)
#         feature_dim = 512  # <-- change after checking output

#         self.head = nn.Sequential(
#             nn.Linear(feature_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         z = self.encoder(x)   # (B, feature_dim)
#         return self.head(z)


def main():

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)


    # ------------------------------------------------- Load data -------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------

    data_dir = "/projects/net_contrast_classification/contrast_phase/Preprocessing/Contrast_data/preprocessed_data.csv"

    batch_size = 2
    epochs = 50

    model_map = {0: "ResNet10", 1: "CNN8"
                # , 2: "Merlin"
                }
    
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    model_name = model_map[task_id]


    train_dataset = PTDataset(data_dir, split = "train", label_name="contrast", encoder=None, add_augmented=False)
    le = train_dataset.le
    val_dataset = PTDataset(data_dir, split = "val", label_name="contrast", encoder=le)
    test_dataset = PTDataset(data_dir, split = "test", label_name="contrast", encoder=le)


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
        model = CNN8(num_classes=len(le.classes_), dropout_rate=0.2)

    # elif model_name == "ViT":
    #     model = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True, dropout_rate = 0.3, save_attn = True)
    
    # else:
    #     model = MerlinModel(num_classes=len(le.classes_), dropout_rate=0.2)


    print(f"\nTraining {model_name} for contrast phase classification\n")

    save_path = "/projects/net_contrast_classification/contrast_phase/DeepLClassifiers/Contrast"
    os.makedirs(save_path, exist_ok=True)

    trained_model = train_cnn(model,
                                    train_loader,
                                    weights = None,
                                    val_loader=val_loader,
                                    epochs = epochs,
                                    early_stopping=10,
                                    save_path = save_path
                                    )
    


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