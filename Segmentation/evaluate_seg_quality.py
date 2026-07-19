import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.append("/projects/net_contrast_classification/contrast_phase")
from Segmentation.cross_segment_models import AttentionFusionSegmentation
from Segmentation.train_cross_seg_patches import SegDataset
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import Compose, EnsureTyped
from monai.inferers import sliding_window_inference



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------
    # Load test data
    # -------------------------------

    data_dir = "/projects/net_contrast_classification/contrast_phase/Preprocessing/Segmentation_data/full_pairs.csv"
    df = pd.read_csv(data_dir)

    patch_size = (64,64,64)

    transform = Compose([EnsureTyped(keys=["image", "label"])])

    test_dataset = SegDataset(df=df,split="test",transform=transform)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

    # -------------------------------
    # Load model
    # -------------------------------

    model = AttentionFusionSegmentation()
    checkpoint = torch.load("/projects/net_contrast_classification/contrast_phase/Segmentation/AttentionFusionSegmentation.pth",map_location=device)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    results = []

    # -------------------------------
    # Evaluation
    # -------------------------------

    with torch.no_grad():

        for batch in tqdm(test_loader):

            image = batch["image"].to(device)
            lesion = batch["label"].to(device)

            # sliding window expects full volume input

            def predictor(x_in):
                a = x_in[:, 0:1]
                p = x_in[:, 1:2]
                return model(a, p)

            logits = sliding_window_inference(
                inputs=image,
                roi_size=patch_size,
                sw_batch_size=4,
                predictor=predictor,
                overlap=0.5,
                mode="gaussian",
            )

            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).float()

            dice_metric = DiceMetric(include_background=False,reduction="mean")
            iou_metric = MeanIoU(include_background=False)

            dice_metric(y_pred=pred,y=lesion)
            iou_metric(y_pred=pred,y=lesion)

            dice = dice_metric.aggregate().item()
            iou = iou_metric.aggregate().item()

            # reset for next patient
            dice_metric.reset()
            iou_metric.reset()

            results.append({

                "SubjectKeyRadiology": batch["SubjectKeyRadiology"][0],
                "ExamDate": batch["ExamDate"][0],

                "Dice": dice,
                "IoU": iou,

                "GT_voxels": lesion.sum().item(),
                "Pred_voxels": pred.sum().item()

            })


    results_df = pd.DataFrame(results)

    print("\nPer-case performance:")
    print(results_df[["Dice", "IoU"]].describe())

    print("\nMean performance:")
    print(
        results_df[
            [
                "Dice",
                "IoU"
            ]
        ].mean()
    )

    results_df.to_csv("/projects/net_contrast_classification/contrast_phase/Segmentation/seg_test_metrics.csv",index=False)


if __name__ == "__main__":
    main()