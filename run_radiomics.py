import os, sys
import numpy as np
import pandas as pd
from radiomics_pipeline import feature_extract, append_df_to_csv

def main():
    data = pd.read_csv("/projects/net_contrast_classification/contrast_phase/cleaned_data.csv")
    folder_path = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET"
    output_path = "/projects/net_contrast_classification/contrast_phase/features"
    os.makedirs(output_path, exist_ok=True)

    files = data["NiiFile"].tolist()
    phases = data["contrast"].tolist()

    num_batches = 4
    idx_splits = np.array_split(np.arange(len(files)), num_batches)

    if len(sys.argv) != 2:
        print("Usage: python run_radiomics.py <batch_index>")
        sys.exit(1)

    batch_id = int(sys.argv[1]) 
    
    if batch_id < 0 or batch_id >= num_batches:
        print(f"Job index must be between 0 and {num_batches-1}")
        sys.exit(1)

    batch_indices = idx_splits[batch_id].tolist()

    batch_files = [files[i] for i in batch_indices]
    batch_phases = [phases[i] for i in batch_indices]

    output_csv_path = os.path.join(output_path, f"features_batch_{batch_id}.csv")
    wrote_header = False
    processed_images, total_rows = 0, 0 

    for file_path, phase in zip(batch_files, batch_phases):
        df_img = feature_extract(file_path, phase, folder_path)
        wrote_header, processed_images, total_rows = append_df_to_csv(
            df_img, output_csv_path, wrote_header, processed_images, total_rows)


    print(
        f"Job {batch_id} done. Processed {processed_images} images, "
        f"wrote {total_rows} feature rows to {output_csv_path}"
    )


if __name__ == "__main__":
    main()
