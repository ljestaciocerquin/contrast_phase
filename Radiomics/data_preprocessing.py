import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit


def radiomics_load(data_dir, radiomics_dir):
    data = pd.read_csv(data_dir)

    rad = []
    for csv_path in radiomics_dir.glob("vol*/*.csv"):  # picks CSVs from vol1, vol2
        df = pd.read_csv(csv_path)
        rad.append(df)

    radiomics = pd.concat(rad, ignore_index=True)

    radiomics = radiomics[radiomics.contrast != '0']    # remove zero contrast

    radiomics_merged = radiomics.merge(data[['phase_timing', 'MatchKey', 'SubjectKeyRadiology']], left_on = 'image', right_on= 'MatchKey', how = 'left')
    radiomics_merged = radiomics_merged.drop("MatchKey", axis=1)

    col = radiomics_merged.pop('SubjectKeyRadiology')
    radiomics_merged.insert(1, 'SubjectKeyRadiology', col)

    col = radiomics_merged.pop('phase_timing')
    radiomics_merged.insert(2, 'phase_timing', col)


    return radiomics_merged

def train_test_split(data, test_size = 0.3, grouping = "SubjectKeyRadiology"):
    splitter = GroupShuffleSplit(test_size=test_size,random_state=42)

    train_idx, test_idx = next(splitter.split(data, groups=data[grouping]))

    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    return train_data, test_data

def var_and_corr_filtering(
    df,
    feature_cols,
    var_threshold=1e-6,
    corr_threshold=0.8):

    drop_dict = {}

    for organ in sorted(df["organ"].unique()):

        idx = df["organ"] == organ
        organ_df = df.loc[idx, feature_cols]

        # ---------- Variance filtering ----------
        variances = organ_df.var(skipna=True)
        keep_var = variances[variances > var_threshold].index

        organ_var = organ_df[keep_var]

        # ---------- Correlation filtering ----------
        corr = organ_var.corr(method="spearman").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = [
            col for col in upper.columns
            if (upper[col] > corr_threshold).any()
        ]

        drop_dict[organ] = to_drop

    return drop_dict

def expand_features(df, meta_cols = ["image", "contrast", "phase_timing", "SubjectKeyRadiology"]):

    feature_cols = [c for c in df.columns if c not in meta_cols + ["organ"]]

    # Start with only meta columns
    expanded = df[meta_cols].drop_duplicates().copy()

    for organ, organ_df in df.groupby("organ"):
        organ_df = organ_df.copy()
        
        # Prefix features with organ name and remove '_original_'
        organ_feature_cols = {c: f"{organ}_{c}".replace("_original_", "_") for c in feature_cols}
        organ_df = organ_df.rename(columns=organ_feature_cols)
        
        # Keep only meta + prefixed features
        organ_df = organ_df[meta_cols + list(organ_feature_cols.values())]
        
        # Merge into wide dataframe
        expanded = expanded.merge(organ_df, on=meta_cols, how="left").sort_values(by=meta_cols)

    return expanded

def preprocess_train(data, meta_cols, NA_filter_threshold=0.2, pca_components=30):

    data = data.copy()

    feature_cols = [c for c in data.columns if c not in meta_cols]

    # ---------- Variance & correlation filtering ----------
    cols_to_drop = var_and_corr_filtering(
        data,
        feature_cols,
        var_threshold=1e-6,
        corr_threshold=0.8
    )

    for organ, cols in cols_to_drop.items():
        data.loc[data.organ == organ, cols] = np.nan

    # ---------- Remove columns that are fully NA ----------
    data_filtered = data.dropna(how="all", axis=1)

    # ---------- Expand features ----------
    data_expanded = expand_features(data_filtered)

    # ---------- Remove columns with too many NA ----------
    data_expanded = data_expanded.loc[
        :, data_expanded.notna().mean() >= NA_filter_threshold
    ]

    expanded_feature_cols = [c for c in data_expanded.columns if c not in meta_cols]

    X = data_expanded[expanded_feature_cols]

    # ---------- Imputation ----------
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    # ---------- Scaling ----------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # ---------- PCA ----------
    pca = PCA(n_components=pca_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # ---------- Convert scaled features back to dataframe ----------
    X_scaled = pd.DataFrame(
        X_scaled,
        columns=expanded_feature_cols,
        index=data_expanded.index
    )
    meta_cols_subset = [c for c in meta_cols if c != "organ"]
    data_processed = pd.concat(
        [data_expanded[meta_cols_subset], X_scaled],
        axis=1
    )

    return {
        "cols_to_drop": cols_to_drop,
        "expanded_feature_cols": expanded_feature_cols,
        "imputer": imputer,
        "scaler": scaler,
        "pca": pca,
        "data_processed": data_processed,
        "X_pca": X_pca
    }

def preprocess_test(data, meta_cols, pipeline):

    data = data.copy()

    cols_to_drop = pipeline["cols_to_drop"]
    expanded_feature_cols = pipeline["expanded_feature_cols"]
    imputer = pipeline["imputer"]
    scaler = pipeline["scaler"]
    pca = pipeline["pca"]

    # ---------- Apply organ filtering ----------
    for organ, cols in cols_to_drop.items():
        data.loc[data.organ == organ, cols] = np.nan

    # ---------- Remove fully NA columns ----------
    data_filtered = data.dropna(how="all", axis=1)

    # ---------- Expand features ----------
    data_expanded = expand_features(data_filtered)

    # ---------- Keep same columns as training ----------
    data_expanded = data_expanded.reindex(columns=meta_cols + expanded_feature_cols)

    X = data_expanded[expanded_feature_cols]

    # ---------- Impute ----------
    X_imp = imputer.transform(X)

    # ---------- Scale ----------
    X_scaled = scaler.transform(X_imp)

    # ---------- PCA ----------
    X_pca = pca.transform(X_scaled)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=expanded_feature_cols,
        index=data_expanded.index
    )
    
    meta_cols_subset = [c for c in meta_cols if c != "organ"]
    data_processed = pd.concat(
        [data_expanded[meta_cols_subset], X_scaled],
        axis=1
    )

    return data_processed, X_pca

def PCA_formatting(pca_data, data):
    pca_df = pd.DataFrame(
    pca_data, 
    columns=[f"PCA_{i+1}" for i in range(pca_data.shape[1])],
    index=data.index
    )

    return pca_df
