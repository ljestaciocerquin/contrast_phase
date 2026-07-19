
from monai.transforms import LoadImage
import torch
import pandas as pd


def lesion_volume(lesion):
    """
    Returns lesion physical volume in mm3.
    """
    if lesion is None:
        return 0.0

    spacing = lesion.pixdim
    print(f"Spacing: {spacing}")
    voxels = (lesion > 0).float().sum().item()
    volume = voxels * spacing[0] * spacing[1] * spacing[2]

    return volume


def load(path):
    loader = LoadImage(image_only=True, ensure_channel_first=True)
    if path is None or isinstance(path, float):
        return None
    img=loader(path)
    return torch.as_tensor(img)


def load_pair(df, i=0, dataset = "reg"):
    if dataset == "reg":
        row = df.iloc[i]

        a_img = load(row["arterial_image"])
        p_img = load(row["portal_image"])

        a_organs = load(row["arterial_organs"])
        a_liver = (a_organs[0] == 5).unsqueeze(0)
        p_liver = load(row["portal_organs"])

        a_lesion = load(row["arterial_lesion"])
        p_lesion = load(row["portal_lesion"])
        

        return a_img, p_img, a_liver, p_liver, a_lesion, p_lesion
    
    elif dataset == "proc":
        output_path = df.iloc[i, -1]

        data = torch.load(output_path)

        a_img = data["arterial_image"]
        p_img = data["portal_image"]

        a_liver = data["arterial_liver"]
        p_liver = data["portal_liver"]

        a_lesion = data["arterial_lesion"]
        p_lesion = data["portal_lesion"]
        

        return a_img, p_img, a_liver, p_liver, a_lesion, p_lesion
    
    else:
        raise ValueError("Dataset must be eihter \"reg\" for registered or \"proc\" for preprocessed data")
    

def percent_vol_diff(df):

    percent_diffs_a = []
    percent_diffs_p = []

    for i in range(len(df)):
        _, _, _, _, a_lesion0, p_lesion0 = load_pair(df, i, dataset="reg")
        _, _, _, _, a_lesion1, p_lesion1 = load_pair(df, i, dataset="proc")

        a_vol0 = lesion_volume(a_lesion0)
        p_vol0 = lesion_volume(p_lesion0)

        a_vol1 = lesion_volume(a_lesion1)
        p_vol1 = lesion_volume(p_lesion1)

        diff_a = a_vol1 - a_vol0
        diff_p = p_vol1 - p_vol0

        percent_diff_a = diff_a / a_vol0 * 100 if a_vol0 > 0 else 0
        percent_diff_p = diff_p / p_vol0 * 100 if p_vol0 > 0 else 0

        percent_diffs_a.append(percent_diff_a)
        percent_diffs_p.append(percent_diff_p)

    return percent_diffs_a, percent_diffs_p


def main():

    # df_reg = pd.read_csv("/projects/net_contrast_classification/contrast_phase/GuideReg/guidereg/data/registration_pairs.csv")
    df = pd.read_csv("/projects/net_contrast_classification/contrast_phase/Preprocessing/Segmentation_data/full_pairs.csv")
    df = df[df["split"] != "inference"]

    percent_diffs_a, percent_diffs_p = percent_vol_diff(df)

    mean_diff_a = sum(percent_diffs_a) / len(percent_diffs_a)
    mean_diff_p = sum(percent_diffs_p) / len(percent_diffs_p)

    print(f"Mean percent volume difference in mm3 (arterial): {mean_diff_a}")
    print(f"Mean percent volume difference in mm3 (portal): {mean_diff_p}")


if __name__ == "__main__":
    main()
