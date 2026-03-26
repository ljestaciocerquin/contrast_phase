import pandas as pd 
import numpy as np
import os, subprocess
from pathlib import PureWindowsPath
from tqdm import tqdm
import sys
import shutil
import gzip


data = pd.read_csv("/projects/net_contrast_classification/contrast_phase/data/cleaned_data_1.csv")
data= data[data.exist_on_server.isna()] # the second batch of files that were added on the server (3287 scans)

# FOLDER PATHS
folder_path = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/not_on_server"
output_folder = "/mnt/rhea/data_private/IRBd23-231/GEPNETs/ARTINET/not_on_server_organs"
os.makedirs(output_folder, exist_ok=True)

# ALL FILES
files = [os.path.join(folder_path, PureWindowsPath(f).name) for f in data["MatchKey"]]

num_batches = 4
splits = np.array_split(files, num_batches)



# ----------- JOB INDEX FROM COMMAND LINE -----------
if len(sys.argv) != 2:
    print("Usage: python totalseg_organs.py <batch_index>")
    sys.exit(1)

batch_index = int(sys.argv[1])    # treats the 2nd arg as the batch index

if batch_index < 0 or batch_index >= num_batches:
    print(f"Job index must be between 0 and {num_batches-1}")
    sys.exit(1)

files_batch = splits[batch_index]
print(f"Job {batch_index} will process {len(files_batch)} files.")


# ------------------ RUN TotalSegmentator ------------------
def run_baseline(file, output_folder = output_folder):
    file = str(file)  # ensure normal python string
    basename = os.path.basename(file).replace(".nii.gz", "")
    output_basename = os.path.join(output_folder, basename)
    dst = os.path.join(output_folder, basename + ".organs.nii.gz")

    if os.path.exists(dst):
        print(f"Skipping {basename}, already processed")
        return

    print(f"\n\nProcessing: {file}")


    cmd = [
        "/home/k.minkova/miniconda3/envs/class/bin/TotalSegmentator",
        "-i", file,
        "-o", output_basename,
        "-ta", "total",
        "--ml",
        "--device", "gpu",
        "--fast"
    ]

    try:
        print("Running:", " ".join(cmd))

        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # ----------- FIND THE OUTPUT FILE -----------
        ts_output = None
        for ext in [".nii", ".nii.gz"]:
            candidate = output_basename + ext
            if os.path.exists(candidate):
                ts_output = candidate
                break

        if ts_output is None:
            print(f"No segmentation found for {file}")
            if os.path.exists(output_basename):
                shutil.rmtree(output_basename)
            return

        # ----------- COMPRESS IF NEEDED -----------
        if ts_output.endswith(".nii"):  # compress only if plain .nii
            with open(ts_output, 'rb') as f_in, gzip.open(dst, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(ts_output)  # remove original uncompressed file
            print(f"Compressed and saved {dst}")
        else:
            # Already .nii.gz, just rename
            os.rename(ts_output, dst)
            print(f"Saved {dst}")

        # Clean up TS folder if it exists
        if os.path.exists(output_basename) and os.path.isdir(output_basename):
            shutil.rmtree(output_basename)

    except Exception as e:
        print(f"Error processing {file}: {e}")


# ----------- PROCESS FILES -----------
processed_files = 0 
for f in tqdm(files_batch):
    r = run_baseline(f)
    processed_files += 1
    

for f in os.listdir(output_folder):
    if f.endswith(".organs.nii.gz"):
        src = os.path.join(output_folder, f)
        dst = os.path.join(folder_path, f)
        shutil.move(src, dst)
        print(f"Moved {f} → {folder_path}")


# ----------- PRINT JOB ENDING -----------
print(f"Job {batch_index} done. Processed {processed_files} files. Saved to {output_folder}")



