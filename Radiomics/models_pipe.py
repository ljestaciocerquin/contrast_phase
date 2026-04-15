from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from IPython.display import display
from data_preprocessing import preprocess_test, preprocess_train
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold


def train_tree_models(X_train, y_train):
    """
    Train Decision Tree, Random Forest, and XGBoost classifiers.
    Return models and their predictions on X_test.
    """
    models = {}

    # Decision Tree
    DT = DecisionTreeClassifier(max_depth=6, criterion="log_loss", random_state=42)
    DT.fit(X_train, y_train)
    models["DT"] = DT

    # Random Forest
    RF = RandomForestClassifier(n_estimators=500, max_depth=6, criterion="log_loss", random_state=42)
    RF.fit(X_train, y_train)
    models["RF"] = RF

    # Gradient Boosted Trees
    GBT = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        tree_method="hist",
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42
    )
    GBT.fit(X_train, y_train)
    models["GBT"] = GBT

    return models

def evaluate_model(models, X_test, y_test, encoder, model_name = "GBT"):
    """
    Evaluate trained models using accuracy and classification report.
    """

    # Make predictions
    predictions = {name: model.predict(X_test) for name, model in models.items()}
    accuracies = {}

    for name, pred in predictions.items():
        acc = accuracy_score(y_test, pred)
        accuracies[name] = acc
        print(f"Accuracy of {name}: {acc:.5f}")

    # Choose GBT for detailed evaluation (or highest accuracy)
    if model_name is not None:
        y_pred = predictions[model_name]
        print(f"\nClassification report for {model_name}:")
        print(classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_
        ))

        y_test_named = encoder.inverse_transform(y_test)
        pred_named = encoder.inverse_transform(y_pred)

        display(pd.crosstab(
            y_test_named,
            pred_named,
            rownames=["True contrast"],
            colnames=["Predicted contrast"],
            margins=True
        ))

    return predictions, accuracies

def tsne_visual(X_pca,
                y,
                encoder,
                title,
                scatter_size = 25,
                transparency = 0.7,
                misclassifications = False,
                predictions = None):

    if not misclassifications:
        tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", random_state=42)
        X_tsne = tsne.fit_transform(X_pca)


        plt.figure(figsize=(8,6))

        for label in np.unique(y):
            idx = y == label
            
            plt.scatter(
                X_tsne[idx,0],
                X_tsne[idx,1],
                s=scatter_size,                # point size
                alpha=transparency,
                label=encoder.inverse_transform([label])[0]
            )

        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(title)

        plt.legend(title="Contrast Phase")
        plt.show()
    
    else:
        tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", random_state=42)
        X_tsne = tsne.fit_transform(X_pca)

        mis = predictions != y

        plt.figure(figsize=(8,6))

        for label in np.unique(y):
            idx = y == label
            
            plt.scatter(
                X_tsne[idx,0],
                X_tsne[idx,1],
                s=scatter_size,                
                alpha=transparency,
                label=encoder.inverse_transform([label])[0]
            )

        # highlight misclassified
        plt.scatter(
            X_tsne[mis,0],
            X_tsne[mis,1],
            facecolors="none",
            edgecolors="red",
            s=100,
            linewidth=1.5,
            label="Misclassified"
        )

        plt.title("t-SNE of Test Set with Misclassified Samples")
        plt.legend(title="Contrast Phase")
        plt.show()

def extract_misclass(y_test, y_pred, data, encoder, X_test_index, pred_col="pred_contrast"):

    misclassified_idx = np.where(y_pred != y_test)[0]   # Find misclassified positions in the test arrays
    original_idx = X_test_index[misclassified_idx]      # Map back to original data indices
    misclass_data = data.loc[original_idx].copy()                                 
    misclass_data[pred_col] = encoder.inverse_transform(y_pred[misclassified_idx]) 

    return misclass_data

def window_ct(img, center = 50, width = 300):
        low = center - width/2
        high = center + width/2
        return img.clip(low, high)

# def show_misclass(misclass_data, orig_data, samples_per_class=3, label='contrast'):
#     """
#     Display misclassified images with true and predicted labels using
#     'contrast' and 'pred_contrast' columns in misclass_data.

#     misclass_data: pd.DataFrame
#         Must contain 'MatchKey', 'contrast', and 'pred_contrast'.
#     orig_data: pd.DataFrame
#         Original dataframe with 'MatchKey' and 'server_folder'.
#     samples_per_class: int
#         Number of images to show per class.
#     """

#     # Merge folder paths
#     misclass_data = misclass_data.merge(orig_data[['MatchKey', 'server_folder']], on='MatchKey', how='left')

#     # Sample images per class
#     rows = []
#     for c in misclass_data[label].unique():
#         subset = misclass_data[misclass_data[label] == c]
#         rows.append(subset.sample(n=min(samples_per_class, len(subset))))

#     selected_rows = pd.concat(rows).sort_values(label).reset_index(drop=True)

#     file_list = selected_rows.MatchKey.tolist()
#     folder_list = selected_rows.server_folder.tolist()

#     # Load images
#     images, _ = data_load(file_list, folder_list, organ_seg=False)

    

#     # Plot images
#     for _, group in selected_rows.groupby(label):

#         plt.figure(figsize=(15, 5))

#         for j, idx in enumerate(group.index):
#             img = images[idx][0]
#             slice_img = img[:, :, int(img.shape[-1] // 1.3)]

#             plt.subplot(1, len(group), j + 1)
#             plt.imshow(np.rot90(window_ct(slice_img), 1), cmap="gray")

#             # Use contrast and pred_contrast columns
#             true_label = selected_rows.loc[idx, label]
#             pred_label = selected_rows.loc[idx, "pred_contrast"]
#             plt.title(f"True:{true_label}\nPred: {pred_label}")
#             plt.axis("off")

#         plt.show()

# def show_misclass_widget(misclass_data, orig_data, samples_per_class=3, label='contrast'):
#     """
#     Display misclassified images with an interactive slice slider.
#     """

#     # Merge folder paths
#     misclass_data = misclass_data.merge(orig_data[['MatchKey', 'server_folder']], left_on="image", right_on='MatchKey', how='left')

#     # Sample images per class
#     rows = []
#     for c in misclass_data[label].unique():
#         subset = misclass_data[misclass_data[label] == c]
#         rows.append(subset.sample(n=min(samples_per_class, len(subset))))

#     selected_rows = pd.concat(rows).sort_values(label).reset_index(drop=True)

#     file_list = selected_rows.image.tolist()
#     folder_list = selected_rows.server_folder.tolist()

#     # Load images
#     images, _ = data_load(file_list, folder_list, organ_seg=False)

#     max_slices = max(img[0].shape[-1] for img in images)

#     # Create a slice slider widget
#     slice_slider = widgets.IntSlider(
#         value=max_slices // 2,
#         min=0,
#         max=max_slices - 1,
#         step=1,
#         description="Slice"
#     )

#     def update(slice_idx):
#         """Redraw images for selected slice."""
#         for _, group in selected_rows.groupby(label):
#             plt.figure(figsize=(15, 5))

#             for j, idx in enumerate(group.index):
#                 img = images[idx][0]
#                 slice_idx_use = min(slice_idx, img.shape[-1] - 1)
#                 slice_img = img[:, :, slice_idx_use]

#                 plt.subplot(1, len(group), j + 1)
#                 plt.imshow(np.rot90(window_ct(slice_img), 1), cmap="gray")

#                 true_label = selected_rows.loc[idx, label]
#                 pred_label = selected_rows.loc[idx, "pred_contrast"]
#                 plt.title(f"True:{true_label}\nPred: {pred_label}")
#                 plt.axis("off")

#             plt.show()

#     # Use interactive widget
#     widgets.interact(update, slice_idx=slice_slider)

def train_pipe(train_data, test_data, meta_cols, class_name = "contrast", model_name="GBT", pca=False):

    encoder = LabelEncoder()

    train_pipeline = preprocess_train(train_data, meta_cols)

    train_processed = train_pipeline["data_processed"]
    test_processed, X_test_pca = preprocess_test(test_data, meta_cols, train_pipeline)

    feature_cols = train_processed.columns.difference(meta_cols + ["contrast"])

    if not pca:
        X_train = train_processed[feature_cols]
        X_test = test_processed[feature_cols]
    else:
        X_train = train_pipeline["X_pca"]
        X_test = X_test_pca

    y_train = encoder.fit_transform(train_processed[class_name])
    y_test = encoder.transform(test_processed[class_name])

    models = train_tree_models(X_train, y_train)

    predictions, accuracy = evaluate_model(models, X_test, y_test, encoder)

    if model_name is not None:
        predictions = predictions[model_name]
        accuracy = accuracy[model_name]

    return models, predictions, accuracy

def cross_val(data, meta_cols, class_name = "contrast", model_name="GBT", k=5, grouping="SubjectKeyRadiology", pca=False):

    groups = data[grouping]
    gkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    y = data[class_name]
    acc = []

    for train_idx, test_idx in gkf.split(data, y, groups=groups):

        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        _, _, accuracy = train_pipe(
            train_data, test_data, meta_cols, class_name, model_name, pca
        )

        acc.append(accuracy)

    print(f"{model_name} accuracy: {np.mean(acc):.3f} ± {np.std(acc):.3f}")

    return np.mean(acc)