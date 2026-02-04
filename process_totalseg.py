import pandas as pd 
import numpy as np
import os, subprocess, json
from pathlib import PureWindowsPath
from tqdm import tqdm
import sys


data = pd.read_csv("/projects/net_contrast_classification/contrast_phase/dataset-registration-artinet.csv")

# FOLDER PATHS
folder_path = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET"
output_path = "/projects/net_contrast_classification/contrast_phase/total_seg_output"
os.makedirs(output_path, exist_ok=True)

# ALL FILES
files = [os.path.join(folder_path, PureWindowsPath(f).name) for f in data["NiiFile"]]

num_batches = 4
splits = np.array_split(files, num_batches)



# ----------- JOB INDEX FROM COMMAND LINE -----------
if len(sys.argv) != 2:
    print("Usage: python process_totalseg.py <batch_index>")
    sys.exit(1)

batch_index = int(sys.argv[1])    # treats the 2nd arg as the batch index

if batch_index < 0 or batch_index >= num_batches:
    print(f"Job index must be between 0 and {num_batches-1}")
    sys.exit(1)

files_batch = splits[batch_index]
print(f"Job {batch_index} will process {len(files_batch)} files.")




# ----------- RUN TotalSegmentator -----------
def run_baseline(file):
    basename = os.path.basename(file).replace(".nii.gz", "_phase.json")
    output_file = os.path.join(output_path, basename)

    try:
        subprocess.run(["totalseg_get_phase", "-i", file, "-o", output_file], check=True)

        # load the JSON output
        with open(output_file) as jf:
            contrast_data = json.load(jf)

        # Add filename for reference
        contrast_data["file"] = os.path.basename(file)
        return contrast_data
    
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None
    



# ----------- PROCESS FILES -----------
results = []
for f in tqdm(files_batch):
    r = run_baseline(f)
    if r:
        results.append(r)




# ----------- SAVE RESULTS -----------
output_csv = os.path.join(output_path, f"contrast_phase_results_batch{batch_index}.csv")
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Job {batch_index} done. Processed {len(df)} files. Saved to {output_csv}")