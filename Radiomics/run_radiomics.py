import os, sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from radiomics_pipeline import process_single_image,append_df_to_csv


def main():
    data = pd.read_csv("/projects/net_contrast_classification/contrast_phase/data/cleaned_data_1.csv")
    data= data[data.exist_on_server.notna()] # the batch of files that were initially processed (5447 scans) 

    folder_path = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET"
    output_path = "/projects/net_contrast_classification/contrast_phase/Radiomics/features/vol1"
    os.makedirs(output_path, exist_ok=True)

    files = data["MatchKey"].tolist()
    phases = data["contrast"].tolist()

    num_batches = 4
    idx_splits = np.array_split(np.arange(len(files)), num_batches)

    if len(sys.argv) != 2:
        print("Usage: python run_radiomics.py <batch_index>")
        sys.exit(1)

    batch_id = int(sys.argv[1])
    if batch_id < 0 or batch_id >= num_batches:
        raise ValueError(f"Batch index must be 0–{num_batches-1}")

    batch_indices = idx_splits[batch_id].tolist()

    batch_files = [files[i] for i in batch_indices]
    batch_phases = [phases[i] for i in batch_indices]

    # Number of CPUs available (Slurm-aware)
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    n_workers = min(n_cpus, len(batch_files))

    print(f"Batch {batch_id}: using {n_workers} workers")

    args = [
        (file_path, phase, folder_path)
        for file_path, phase in zip(batch_files, batch_phases)
    ]

    output_csv_path = os.path.join(
        output_path, f"features_batch_{batch_id}.csv"
    )

    wrote_header = False
    processed_images = 0
    total_rows = 0

    with Pool(processes=n_workers) as pool:
        for df_img in pool.imap_unordered(process_single_image, args, chunksize=1):
            wrote_header, processed_images, total_rows = append_df_to_csv(
                df_img,
                output_csv_path,
                wrote_header,
                processed_images,
                total_rows,
            )

    print(
        f"Job {batch_id} done. "
        f"Processed {processed_images} images, "
        f"wrote {total_rows} rows to {output_csv_path}"
    )


if __name__ == "__main__":
    main()
