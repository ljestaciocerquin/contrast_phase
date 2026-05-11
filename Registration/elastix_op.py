import SimpleITK as sitk
#!/usr/bin/env python3
from __future__ import annotations
import yaml
import gc
import numpy as np
import logging
import traceback
import argparse
import pandas as pd
from pathlib import Path
from typing  import Optional, Tuple, Dict, Any, List
from collections import Counter
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
import os
 
# # Using Dask for parallel processing
# import dask.bag as db
# from dask.diagnostics import ProgressBar

# Imports of the project itself
# from dataset_filter.utils    import read_file
# from utils.file_operations   import load_config
# from utils.logging_utils     import setup_logging

file_exist = os.path.exists

# dummy comment to resolve github issue

def load_liver_mask(mask_path: Path, liver_index: int = 5) -> sitk.Image:
    """
    Load a multi-channel organ mask and extract only the liver channel.

    Expected input shapes:
        [C, H, W, D]  or  [H, W, D]

    liver_index:
        Index of liver channel in the multi-channel mask.
    """

    # Read segmentation
    seg_img = sitk.ReadImage(str(mask_path))

    # Convert to numpy
    seg_np = sitk.GetArrayFromImage(seg_img)

    # Handle shapes
    # SimpleITK returns arrays as [z,y,x] for scalar
    # or [c,z,y,x] for vector images depending on storage
    if seg_np.ndim == 4:
        liver_np = seg_np[liver_index]
    elif seg_np.ndim == 3:
        # Already single-label mask
        liver_np = (seg_np == liver_index).astype(np.uint8)
    else:
        raise ValueError(f"Unexpected mask shape: {seg_np.shape}")

    # Ensure binary
    liver_np = (liver_np > 0).astype(np.uint8)

    # Back to SITK
    liver_img = sitk.GetImageFromArray(liver_np)

    # Copy metadata
    liver_img.CopyInformation(seg_img)

    return sitk.Cast(liver_img, sitk.sitkUInt8)

def _read_image(path: Path, pixel_id: Optional[int] = None) -> sitk.Image:
    """
    Read an image with SimpleITK. Optionally cast to a pixel type.
    Raises with a helpful message on failure.
    """
    try:
        img = sitk.ReadImage(str(path))
        if pixel_id is not None:
            img = sitk.Cast(img, pixel_id)
        return img
    except Exception as e:
        raise RuntimeError(f"Failed to read image: {path}") from e


def _write_image(img: sitk.Image, path: Path, pixel_id: Optional[int] = None) -> None:
    """
    Write an image with SimpleITK. Optionally cast to a pixel type.
    Raises with a helpful message on failure.

    Parameters
    ----------
    img : sitk.Image
        The SimpleITK image to save.
    path : Path
        Destination file path.
    pixel_id : Optional[int], default=None
        If provided, the image will be cast to this pixel type before writing.
        Example: sitk.sitkUInt8, sitk.sitkFloat32
    """
    try:
        if pixel_id is not None:
            img = sitk.Cast(img, pixel_id)
        sitk.WriteImage(img, str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to write image to: {path}") from e


def configure_initial_registration():
    initial_parameter_map = sitk.GetDefaultParameterMap("rigid")
    initial_parameter_map["AutomaticTransformInitialization"]       = ["true"]
    initial_parameter_map["MaximumNumberOfIterations"]              = ["500"]
    initial_parameter_map["AutomaticTransformInitializationMethod"] = ["CenterOfGravity"]
    initial_parameter_map["Metric"]                                 = ["AdvancedMeanSquares"]
    return initial_parameter_map


def configure_rigid_registration(pixel_value):
    rigid_param_map = sitk.GetDefaultParameterMap("rigid")
    rigid_param_map["Transform"]                              = ["EulerTransform"]
    rigid_param_map["NumberOfResolutions"]                    = ["3"]
    rigid_param_map["MaximumNumberOfIterations"]              = ["500"]
    rigid_param_map["Metric"]                                 = ["AdvancedMattesMutualInformation"]
    rigid_param_map["Optimizer"]                              = ["AdaptiveStochasticGradientDescent"]
    rigid_param_map["DefaultPixelValue"]                      = [pixel_value]
    rigid_param_map["ErodeFixedMask"]                         = ["false"]
    rigid_param_map["ErodeMovingMask"]                        = ["false"]
    return rigid_param_map


def _validate_inputs(fixed_img_path, moving_img_path, fixed_mask_path, moving_mask_path):
    if not fixed_img_path.exists() or not moving_img_path.exists():
        msg = f"Fixed scan {fixed_img_path} and/or moving scan {moving_img_path} could not be found!"
        logging.error(msg)
        raise FileNotFoundError(msg)
    if not fixed_mask_path.exists() or not moving_mask_path.exists():
        msg = f"Fixed mask {fixed_mask_path} and/or moving mask {moving_mask_path} could not be found!"
        logging.error(msg)
        raise FileNotFoundError(msg)


def _prepare_directories(out_dir_path, out_dir_T1_path):
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_dir_T1_path.mkdir(parents=True, exist_ok=True)


def _init_log(out_dir_T1_path, idx, fixed_mask_path, moving_mask_path):
    log_file = out_dir_T1_path / f"elastix.log"
    with open(log_file, "a") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"[START_PAIR] idx={idx}\n")
        f.write(f"elastix_dir: {out_dir_T1_path}\n")
        f.write(f"Fixed:  {fixed_mask_path}\n")
        f.write(f"Moving: {moving_mask_path}\n")
        f.write(f"Time:   {pd.Timestamp.now()}\n")
        f.write("="*80 + "\n")
    return log_file


def _finalize_log(log_file, idx):
    with open(log_file, "a") as f:
        f.write("\n" + "-"*80 + "\n")
        f.write(f"[END_PAIR] idx={idx} finished at {pd.Timestamp.now()}\n")
        f.write("-"*80 + "\n")
        

def _dilate_mask(mask, radius):
    return sitk.BinaryDilate(mask, radius, sitk.sitkBall)


def _get_L1_DIL_and_make_soft_mask(fixed_mask_dil, moving_mask_dil, fixed_mask, T0_vec, T1_vec):
    L1_DIL = get_segmentation_after_T1(T0_vec, T1_vec, moving_mask_dil)
    L1_D = sitk.Resample(L1_DIL, fixed_mask)
    liver_union = sitk.Or(fixed_mask_dil, L1_D)
    soft_mask = sitk.SmoothingRecursiveGaussian(liver_union, 0.5)
    return sitk.Cast(soft_mask, sitk.sitkFloat32), L1_DIL, liver_union

def _crop_to_roi(mask_union):
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(mask_union)

    if not label_stats.GetLabels():
        raise ValueError("ROI mask is empty — cannot crop.")

    bbox = label_stats.GetBoundingBox(label_stats.GetLabels()[0])  # (x,y,z,dx,dy,dz)

    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetSize(bbox[3:])
    roi_filter.SetIndex(bbox[0:3])
    return roi_filter

        
def guess_background_value(img: sitk.Image, rim: int = 3) -> int:
    a = sitk.GetArrayFromImage(img)  # z,y,x
    z,y,x = a.shape
    edges = np.concatenate([
        a[:rim,:,:].ravel(), a[-rim:,:,:].ravel(),
        a[:, :rim,:].ravel(), a[:, -rim:,:].ravel(),
        a[:, :, :rim].ravel(), a[:, :, -rim:].ravel()
    ])
    return int(Counter(edges.tolist()).most_common(1)[0][0])


def crop_to_mask(image, mask, margin=0):
    """
    Crop image to the bounding box of the mask.
    
    Args:
        image (sitk.Image): The image to crop.
        mask (sitk.Image): Binary mask (non-zero = ROI).
        margin (int): Optional padding around the bounding box.
    Returns:
        sitk.Image: Cropped image.
    """
    # Label connected components in the mask
    label = sitk.ConnectedComponent(mask > 0)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(label)

    if stats.GetNumberOfLabels() == 0:
        raise ValueError("Mask is empty, cannot crop.")

    # Get bounding box of the largest label (you could loop if needed)
    bbox = stats.GetBoundingBox(1)  # (x, y, z, sizeX, sizeY, sizeZ)

    # Add margin
    start = [max(0, bbox[i] - margin) for i in range(len(bbox)//2)]
    size = [min(image.GetSize()[i] - start[i], bbox[i+3] + 2*margin) for i in range(len(bbox)//2)]

    # Extract ROI
    roi = sitk.RegionOfInterest(image, size, start)
    return roi


# Initial Transformation
def rigid_registration(
        fixed_img:    sitk.Image,
        moving_img:   sitk.Image,
        fixed_mask:    sitk.Image,
        moving_mask:   sitk.Image,
        T0_param_map:  Path,
        pixel_value:   int,
        output_folder: Path
    ) -> Tuple[List[sitk.ParameterMap], sitk.Image]:
    """
    Run a (rigid) initial registration with SimpleElastix, writing its logs to output_folder.
    Returns (transform_parameter_map_vector, pre_aligned_image).
    Raises RuntimeError on failure.
    """
    #import pdb as pd; pd.set_trace()
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(fixed_img)
    elastix.SetMovingImage(moving_img)
    elastix.SetFixedMask(fixed_mask)
    elastix.SetMovingMask(moving_mask)

    param_map = configure_rigid_registration(str(pixel_value))
    if file_exist(str(T0_param_map)):
        elastix.SetInitialTransformParameterFileName(str(T0_param_map))
        T0 = sitk.ReadParameterFile(str(T0_param_map))
    else:
        error_message = f'The initial transform file {T0_param_map} could not be found!'
        logging.error(error_message)
        raise FileNotFoundError(error_message)
    
    elastix.SetParameterMap([param_map])
    
    # Make elastix use only one 1 cpu thread
    elastix.SetNumberOfThreads(1) 
    
    # Ensure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)
    log_file = output_folder / "elastix.log"

    # Output/log setup: SimpleElastix requires a string path
    elastix.LogToFileOn()
    elastix.SetOutputDirectory(str(output_folder))

    logging.debug(f"[{output_folder}] Starting Elastix initial registration...")
    elastix.Execute()
    logging.debug(f"[{output_folder}] Elastix initial registration finished.")
    
    aligned_image   = elastix.GetResultImage()
    T1 = elastix.GetTransformParameterMap()  # vector of maps
    return aligned_image, T1, T0, log_file

def get_segmentation_after_T1(
    T0_map: List[sitk.ParameterMap],
    T1_map: List[sitk.ParameterMap],
    moving_seg: sitk.Image,
) -> sitk.Image:
    """
    Resample a moving segmentation using the final stage of the initial transform.
    Uses nearest-neighbor (FinalBSplineInterpolationOrder = 0) to keep labels crisp.
    """
    if not T0_map or not T1_map:
        raise ValueError("Empty transform parameter map vector.")

    # Use the last stage (usually the final result)
    T1_map = T1_map[-1]
    T1_map["FinalBSplineInterpolationOrder"] = ["0"]
    T1_map["DefaultPixelValue"]              = ["0"]

    transformix = sitk.TransformixImageFilter()
    transformix.SetTransformParameterMap(T0_map)
    transformix.AddTransformParameterMap(T1_map) 
    transformix.SetMovingImage(moving_seg)
    transformix.Execute()

    warped = transformix.GetResultImage()
    return sitk.Cast(warped, sitk.sitkUInt8)



# Function to register a pair
def _process_one_pair(
    fixed_img_path:         str,
    moving_img_path:        str,
    fixed_mask_path:        str,
    moving_mask_path:       str,
    fixed_tumor_path:       str,
    moving_tumor_path:      str,
    T0_file_path:           str,
    T1_file_path:           str,
    radius_to_dilate:       int,
    output_directory:       str,
    output_directory_T1:    str,
    # is_test_data:           bool,
    save_scans:             bool,
    idx:                    Optional[int] = None,
) -> Dict[str, Any]:
    # Return a dict with status and metadata to save
    
    fixed_img_path     = Path(fixed_img_path)
    moving_img_path    = Path(moving_img_path)
    fixed_mask_path    = Path(fixed_mask_path)
    moving_mask_path   = Path(moving_mask_path)
    fixed_tumor_path   = Path(fixed_tumor_path)  if fixed_tumor_path else None
    moving_tumor_path  = Path(moving_tumor_path) if moving_tumor_path else None
    T0_file_path       = Path(T0_file_path)
    
    
    out_dir_path    = Path(output_directory)
    out_dir_T1_path = Path(output_directory_T1)
    
    try:
        # Step 1. Validate, ensure directories exist and start logging of the elastix.log
        _validate_inputs(fixed_img_path, moving_img_path, fixed_mask_path, moving_mask_path)
        _prepare_directories(out_dir_path, out_dir_T1_path)
        log_file = _init_log(out_dir_T1_path, idx, fixed_mask_path, moving_mask_path)
        
        # Step 2. Load images
        fixed_image  = _read_image(fixed_img_path)
        moving_image = _read_image(moving_img_path)
        fixed_mask  = load_liver_mask(fixed_mask_path, liver_index=5)
        moving_mask = load_liver_mask(moving_mask_path, liver_index=5)

        if fixed_tumor_path and fixed_tumor_path.exists() and moving_tumor_path and moving_tumor_path.exists(): # set to None since not needed for my task
            fixed_tumor  = _read_image(fixed_tumor_path, sitk.sitkUInt8)
            moving_tumor = _read_image(moving_tumor_path,sitk.sitkUInt8)
        
        # Step 3. Dilate masks
        fixed_mask_dil = _dilate_mask(fixed_mask, radius_to_dilate)
        moving_mask_dil = _dilate_mask(moving_mask, radius_to_dilate)

        # Step 4. Rigid registration
        # Identifying the default pixel value
        pixel_value = guess_background_value(fixed_image)
        I1, T1_vec, T0_vec, log_file = rigid_registration(
            fixed_image, 
            moving_image, 
            fixed_mask_dil, 
            moving_mask_dil, 
            T0_file_path, 
            pixel_value, 
            out_dir_T1_path)

        
        # Save transformation parameters
        t1_file = Path(T1_file_path)
        t1_file.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteParameterFile(T1_vec[-1], str(t1_file))
        logging.info('Wrote final parameter map -> %s', str(t1_file))
        
        # Save end marker
        _finalize_log(log_file, idx)


        # -------- if make_masked_images end -----------

        # Step 5. Save masks
        if save_scans:
            
            L1 = get_segmentation_after_T1(T0_vec, T1_vec, moving_mask)
            
            filename_F  = out_dir_path / "F.nii.gz"
            # filename_M  = out_dir_path / "M.nii.gz"
            # filename_LF  = out_dir_path / 'L_fix.nii.gz'
            # filename_LM  = out_dir_path / 'L_mov.nii.gz'
            
            filename_I1 = out_dir_path / "I1.nii.gz"
            filename_L1_DIL = out_dir_path / 'L1_DIL.nii.gz'
            filename_L1 = out_dir_path / "L1.nii.gz"

            # filename_NF  = out_dir_path / 'NewFixedImage.nii.gz'
            filename_NM  = out_dir_path / 'NewMovingImage.nii.gz'
            filename_LDF = out_dir_path / 'LD_fix.nii.gz'
            #filename_LDM = out_dir_path / 'LD_mov.nii.gz'
            #filename_LU  = out_dir / 'LU_1.nii.gz'
            #filename_LG  = out_dir / 'LG_2.nii.gz'

            _write_image(fixed_image,  str(filename_F), sitk.sitkInt16)
            _write_image(moving_image, str(filename_NM), sitk.sitkInt16)
            _write_image(I1, str(filename_I1), sitk.sitkInt16)
            _write_image(L1, str(filename_L1), sitk.sitkUInt8)


            # if is_test_data:
            #     N1    = get_segmentation_after_T1(T0_vec, T1_vec, moving_tumor)
            #     N_fix_c = roi_filter.Execute(fixed_tumor)
            #     N1_c    = roi_filter.Execute(N1)
            #     filename_N_fix_c  = out_dir_path / 'N_fix.nii.gz'
            #     filename_N1_c     = out_dir_path / 'N1.nii.gz'

            #     _write_image(N_fix_c, str(filename_N_fix_c), sitk.sitkUInt8)
            #     _write_image(N1_c,    str(filename_N1_c), sitk.sitkUInt8)

            #     #N_mov_c = roi_filter.Execute(moving_tumor)
            #     #filename_N_mov_c  = out_dir_path / 'N_mov.nii.gz'
            #     #_write_image(N_mov_c, str(filename_N_mov_c), sitk.sitkUInt8)
            
            
        
        return{
            'ok':          True,
            'in_fix':      fixed_img_path,
            'in_mov':      moving_img_path,
            'out_dir':     output_directory,
            't0_file':     str(t1_file),
            'log_file':    str(log_file)
            
        }
        
    except Exception as e:
        logging.error(f"Error processing pair ({fixed_img_path}, {moving_img_path}): {repr(e)}")
        return{
            'ok':     False,
            'in_fix': fixed_img_path,
            'in_mov': moving_img_path,
            'error': traceback.format_exc()
        }
    
    finally:
        # --- Free memory aggressively ---
        for var in ["fixed_image", "moving_image", "fixed_mask", "moving_mask", "aligned_mask", "T0_vec"]:
            if var in locals():
                del locals()[var]
        gc.collect()

def setup_logging(args):
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

def load_config(yaml_path: Path, config_to_read: str) -> Dict[str, Any]:
    with open(yaml_path, "r") as f:
        cfg_all = yaml.safe_load(f) or {}
    cfg = cfg_all.get(config_to_read, cfg_all)
    return cfg

def read_file(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Reads .csv, .xlsx, .xls, or .txt tabular data file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        return pd.read_csv(filepath, **kwargs)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(filepath, **kwargs)
    elif ext == '.txt':
        return pd.read_csv(filepath, sep='\t', **kwargs)
    elif ext == 'pkl':
        pd.read_pickle(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel largest-components cleanup")
    parser.add_argument("--config",        required=True,              help="Path to YAML config")
    parser.add_argument("-v", "--verbose", action="count", default=1,  help="Increase verbosity (-v, -vv)")
    parser.add_argument("--workers",       type=int,       default=4, help="Parallel workers (processes)")
    parser.add_argument("--debug",         type=int, default=0,        help="Run in debug mode with only N cases")
    parser.add_argument("--dry-run",       action="store_true",        help="Only list jobs, do not execute")
    parser.add_argument("--subject", type=str, help="Run only one subject ID")
    args = parser.parse_args()

    setup_logging(args.verbose)
    cfg = load_config(Path(args.config), "rigid_registration")

    input_file      = Path(cfg["input_file"])
    out_dir         = Path(cfg["output_folder"])
    out_dir_T1      = Path(cfg["output_folder_T1"])
    T0_file_name    = cfg["input_file_param"]
    T1_file_name    = cfg["output_file_param"]
    raw_data_dir    = cfg["folder_to_change"]
    proc_data_dir   = cfg["new_folder"]
    fixed_img_col  = cfg["fix_img_col"]
    moving_img_col = cfg["mov_img_col"]
    fixed_mask_col  = cfg["fix_liv_col"]
    moving_mask_col = cfg["mov_liv_col"]
    fixed_tumor_col  = cfg["fix_tum_col"]
    moving_tumor_col = cfg["mov_tum_col"]
    subject_id_col  = cfg["subject_col"]
    scan_dates_col  = cfg["scan_dates"]
    radius_to_dilate= cfg["radius"]
    is_test_data    = cfg.get("is_test_data", False)
    save_scans      = cfg.get("save_scans", False)
    workers         = args.workers if args.workers else cfg.get("workers", 20)
    mode            = cfg.get("mode", "local")  # "local" or "slurm"

    if not input_file.exists():
        error_message = f"Input file not found: {input_file}"
        logging.error(error_message)
        raise FileNotFoundError(error_message)

    df_all = read_file(str(input_file))
    df     = df_all.drop_duplicates(subset=[subject_id_col], keep="first")
    
    if args.subject:
        #  python -m registration_classical.mask_based_initial_registration --config /projects/liver_image_registration/GEPNET-Tracer/configs/config_registration_classical.yaml --subject NKI-d23231-00-0566
        df = df[df[subject_id_col] == args.subject]
        logging.warning("Restricted run: subject %s (%d rows)", args.subject, len(df))
        for _, row in df.iterrows():
            fi = row["fixed_image"]
            mi = row["moving_image"]
            fl = str(row["fixed_organs"]).replace(raw_data_dir, proc_data_dir) # row["fixed_liver_seg"]  #
            ml = str(row["moving_organs"]).replace(raw_data_dir, proc_data_dir) # row["moving_liver_seg"] #
            ft = row["fixed_lesion_seg"]
            mt = row["moving_lesion_seg"]
            out_dir = Path(cfg["output_folder"]) / row["SubjectKeyRadiology"] / row["ExamDate"]
            out_dir_T1 = out_dir / cfg["output_folder_T1"]
            out_dir_T0 = Path(str(out_dir_T1).replace('T1', 'T0'))
            t1_file = out_dir_T1 / cfg["output_file_param"]
            t0_file = out_dir_T0 / cfg["input_file_param"]
            

            _ = _process_one_pair(fi, mi,
                                  fl, ml,
                                  ft, mt,
                                  str(t0_file), str(t1_file), 
                                  cfg["radius"],
                                  str(out_dir), str(out_dir_T1),
                                  cfg["is_test_data"],
                                  cfg["save_scans"], 0)
            
        logging.info("Processed %d pairs: ", len(df))
        logging.info("Results saved to %s", out_dir)
        return
            
    # Debug mode → restrict to first N rows
    if args.debug > 0:
        logging.warning(f"DEBUG MODE: limiting to first {args.debug} rows")
        df = df.head(args.debug)
        
        
    # Dry run → just log planned jobs
    if args.dry_run:
        logging.warning("DRY RUN: No jobs will be executed, just listing planned jobs")
        for _, row in df.iterrows():
            fi = row[fixed_img_col]
            mi = row[moving_img_col]
            fl   = str(row[fixed_mask_col]).replace(raw_data_dir, proc_data_dir)
            ml   = str(row[moving_mask_col]).replace(raw_data_dir, proc_data_dir)
            ft = row[fixed_tumor_col]
            mt = row[moving_tumor_col]
            subj = row[subject_id_col]
            date = row[scan_dates_col]
            out_dir_pair = out_dir / subj / date
            logging.info(f"[DRY-RUN] Would process FIXED_IMAGE={fi}, MOVING_IMAGE={mi}, FIXED_MASK={fl}, MOVING_MASK={ml}, FIXED_TUMOR={ft}, MOVING_TUMOR={mt}, OUT={out_dir_pair}")
        return  # exit early


if __name__ == "__main__":
    main()
