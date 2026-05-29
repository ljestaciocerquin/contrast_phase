import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from IPython.display import display
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from data_preprocessing import preprocess_test, preprocess_train
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from data_preprocessing import radiomics_load, train_test_split, preprocess_train, preprocess_test


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

def evaluate_model(model, X_test, y_test, encoder):
    """
    Evaluate trained models using accuracy and classification report.
    """

    # Make predictions
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy of {type(model).__name__}: {acc:.5f}")

    print(f"\nClassification report for {type(model).__name__}:", flush=True)
    print(classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_
        ), flush=True)

    y_test_named = encoder.inverse_transform(y_test)
    pred_named = encoder.inverse_transform(y_pred)

    display(pd.crosstab(
            y_test_named,
            pred_named,
            rownames=["True contrast"],
            colnames=["Predicted contrast"],
            margins=True
        ), flush=True)

    return y_pred, acc

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


def main():
    data_dir = Path("/projects/net_contrast_classification/contrast_phase/data/cleaned_data_1.csv")
    features_dir = Path("/projects/net_contrast_classification/contrast_phase/Radiomics/features")
    contrast_le = LabelEncoder()

    model_names = ["DT", "RF", "GBT"]

    radiomics = radiomics_load(data_dir, features_dir)
    radiomics = radiomics[radiomics['SubjectKeyRadiology'].notna()] # ensure no missing values in grouping column

    train_data, test_data = train_test_split(radiomics, test_size=0.3, grouping="SubjectKeyRadiology")

    meta_cols = list(train_data.iloc[:, :5].columns)

    train_pipeline = preprocess_train(train_data, meta_cols, NA_filter_threshold=0.2, pca_components=30)

    train_processed = train_pipeline["data_processed"]

    X_train_pca = train_pipeline["X_pca"]

    test_processed, X_test_pca = preprocess_test(test_data, meta_cols,train_pipeline)

    X_train, y_train = train_processed.iloc[:, 4:], contrast_le.fit_transform(train_processed.contrast)
    X_test, y_test = test_processed.iloc[:, 4:], contrast_le.fit_transform(test_processed.contrast)

    print(f"Shape of train data: {X_train.shape}\nShape of train labels: {y_train.shape}", flush=True)
    print(f"Shape of test data: {X_test.shape}\nShape of test labels: {y_test.shape}", flush = True)

    models = train_tree_models(X_train, y_train)
    for model in model_names:
        print(f"Evaluating {model}...", flush=True)
        evaluate_model(models[model], X_test, y_test, contrast_le)




if __name__ == "__main__":
    main()