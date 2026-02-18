import os, sys
import numpy as np
import pandas as pd
from radiomics_pipeline import feature_extract, save_results

def main():
    data = pd.read_csv("/projects/net_contrast_classification/contrast_phase/dataset-registration-artinet.csv")
    folder_path = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET"
    output_path = "/projects/net_contrast_classification/contrast_phase/features"
    os.makedirs(output_path, exist_ok=True)

    files = data["NiiFile"].tolist()
    phases = data["contrast"].tolist()

    num_batches = 4
    idx_splits = np.array_split(np.arange(len(files)), num_batches)

    # # batch_id = None
    # if len(sys.argv) > 1:
    #     batch_id = int(sys.argv[1])  # treats the 2nd arg as the batch index
    # elif "SLURM_ARRAY_TASK_ID" in os.environ:
    #     batch_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    # else:
    #     batch_id = 0

    # if batch_id >= num_batches:
    #     print(f"Batch {batch_id} out of range, exiting.")
    #     return

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

    df = feature_extract(batch_files, batch_phases, folder_path)
    output_csv_path = os.path.join(output_path, f"features_batch_{batch_id}.csv")
    save_results(df, output_csv_path)
    print(f"Job {batch_id} done. Processed {len(df)} files. Saved to {output_csv_path}")


if __name__ == "__main__":
    main()
