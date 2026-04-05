from monai.transforms import LoadImage
import pandas as pd 
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from pathlib import PureWindowsPath
import os
from pathlib import Path


def list_organs(selected=None, by='name'):
    """
    List all organs segmented by TS either by name or by id

    selected = None: list of str or int
                A list with all organs you want to segment; by default returns all specified (contrast relevant) organs 
    by = 'name': str 
        Specifies the organ identification method - either by name or by id

    returns organ_names (list), organ_ids (list)
    """


    ORGANS = {
        "spleen": 1,
        "kidney_right": 2,	
        "kidney_left": 3, 	
        "gallbladder": 4, 	
        "liver": 5,	
        "stomach": 6, 
        "pancreas": 7,  	
        "adrenal_gland_right": 8,
        "adrenal_gland_left": 9,
        "small_bowel": 13,
        "heart": 51,
        "aorta": 52,
        "pulmonary_vein": 53,
        "superior_vena_cava": 62,
        "inferior_vena_cava": 63,
        "portal_vein_and_splenic_vein": 64,
        "iliac_artery_left": 65,
        "iliac_artery_right": 66,
        "iliac_vena_left": 67,
        "iliac_vena_right": 68
    }

    if selected is None:
        selected_items = list(ORGANS.items())

    else:
        if isinstance(selected, (str, int)):
            selected = [selected]

        if by =='name':
            missing = [name for name in selected if name not in ORGANS]
            if missing:
                raise ValueError(f"Unknown organ name(s):{missing}")
            selected_items = [(name, ORGANS[name]) for name in selected]

        elif by =='id':
            id_to_name = {oid: name for name ,oid in ORGANS.items()}
            missing = [oid for oid in selected if oid not in ORGANS.values()]
            if missing:
                raise ValueError(f"Unknown organ id(s):{missing}")
            selected_items = [(id_to_name[oid], oid) for oid in selected]
        else: 
            raise ValueError("\'by\' must be either \'name\' or \'id\'")
    
    organ_names = [name for name,_ in selected_items]
    organ_ids = [oid for _,oid in selected_items]

    return organ_ids, organ_names

def multi_channel(organ_ids, seg_ct):
    """
    Convert a multi-label mask to a multi-channel mask [num_organs, H, W, D]

    organ_ids: list of int
               The integer labels corresponding to each organ
    seg_ct: np.ndarray 
        The multi-label mask 

    returns channels: np.ndarray of shape [num_organs, H, W, D]
    """
    
    if seg_ct.ndim == 4 and seg_ct.shape[0] == 1:
        seg_ct = seg_ct[0]  # remove singleton channel
    elif seg_ct.ndim == 3:
        pass  # already okay
    else:
        raise ValueError(f"Unexpected shape for seg_ct: {seg_ct.shape}")
    
    channels = np.zeros((len(organ_ids), *seg_ct.shape), dtype=np.uint8)
    for i, oid in enumerate(organ_ids):
        channels[i] = (seg_ct == oid).astype(np.uint8)
        
    return channels
    
def data_load(files, folder_paths, channel_first = True, organ_seg = True, organs = None, by = 'name'):
    """
    Load the CT image files and organ segmentation files (optionally)

    files: list of str
            List of all the file paths from which we want to load the images
    folder_path: list of str
            The folder path where all the CT images/organ segmentations are located
    channel_first: Boolean
            Ensure each loaded image has the channel first, i.e., is of shape [1, H, W, num_slices]
    organ_seg: Boolean 
            Specify whether you want to load organ segmentations 
    organs: list of str or int
            List the ROIs for the radiomics extraction; by default 20 relevant ROIs are used
    by: str
            Specify whether ROIs are listed by their names or by their ids as given by TS

    returns images: np.ndarray, segments: np.ndarray
    """

    images = [] 
    segments = []  
    loader = LoadImage(image_only=True, ensure_channel_first=channel_first)
    
    # Ensure lists and flatten any single-element lists inside
    file_list = files if isinstance(files, list) else [files]
    folder_list = folder_paths if isinstance(folder_paths, list) else [folder_paths]


    for file, folder_path in zip(file_list, folder_list):
        file_dir = os.path.join(folder_path, PureWindowsPath(file).name)
        images.append(loader(file_dir).numpy())

    if organ_seg:
        organ_ids, _ = (list_organs() if organs is None else list_organs(organs, "name" if by == 'name' else 'id'))

        organ_segments = [str(Path(file).with_name(Path(file).name.replace(".nii.gz", ".organs.nii.gz"))) for file in file_list]
        for seg, folder_path in zip(organ_segments, folder_list):
            seg_dir = os.path.join(folder_path, PureWindowsPath(seg).name)
            seg_ct = loader(seg_dir)
            seg_ct = seg_ct.numpy()

            if organ_ids is not None:
                seg_ct = multi_channel(organ_ids, seg_ct) # Convert single-label segmentation -> multi-channel

            segments.append(seg_ct)

    return images, segments

def create_extractor(bin_width = 20 , verbose = False):
    """
    Create the extractor object. By default all first-order features are enabled and 
    a predefined selection of second-order features is hardcoded

    bin_width: int
               The bin width of the gray intensities histogram 

    returns extractor: object
    """

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder', enabled=True)
    extractor.enableFeaturesByName(
        glcm=[
            'Contrast',
            'Correlation',
            'Autocorrelation',
            'ClusterTendency',
            'DifferenceEntropy'
        ],
        glszm=[
            'ZoneEntropy',
            'SmallAreaEmphasis',
            'GrayLevelNonUniformity',
            'SizeZoneNonUniformity'
        ],
        gldm=[
            'SmallDependenceEmphasis',
            'DependenceNonUniformity'
        ]
    )

    # settings
    extractor.settings['binWidth'] = bin_width
    if not verbose:
        extractor.settings['verbose'] = False
    extractor.settings['enableDiagnostics'] = False

    return extractor

def nan_feature_template(extractor):
    """
    Build a template of radiomics output keys initialized to NaN.
    """

    dummy_img = np.random.default_rng(0).random((9, 9, 9), dtype=np.float32)
    dummy_mask = np.zeros((9, 9, 9), dtype=np.uint8)
    dummy_mask[2:7, 2:7, 2:7] = 1  # Fills a centered cube with label 1 to ensure non-empty ROI

    try:
        res = extractor.execute(
            sitk.GetImageFromArray(dummy_img),
            sitk.GetImageFromArray(dummy_mask),
        )
    except Exception:
        return {}

    return {k: np.nan for k in res.keys() if not k.startswith('diagnostics')}

def feature_extract(files, phases, folder_path, channel_first = True, organ_seg = True, organs = None, by = 'name'):
    """
    Convert a single-label mask to a multi-channel mask [num_organs, H, W, D]
    
    files: str or list of str
        Accepts either a string of a single file path or a list of file paths
    phases: str or list of str
        Accepts either a string or a list of strings for the contrast phase; must match with files   
    folder_path: str of Path
        The path where the .nii files are located
    channel_first: Boolean
        Ensure each loaded image has the channel first, i.e., is of shape [1, H, W, num_slices]
    organ_seg: Boolean 
        Specify whether you want to load organ segmentations 
    organs: list of str or int
        List the ROIs for the radiomics extraction; by default 20 relevant ROIs are used
    by: str
        Specify whether ROIs are listed by their names or by their ids as given by TS

    returns df: pd.DataFrame
    """

    all_results = []
    _, organ_names = list_organs(organs, by)
    extractor = create_extractor()
    nan_template = nan_feature_template(extractor)
    
    file_list = files if isinstance(files, list) else [files]
    phases_list = phases if isinstance(phases, list) else [phases]

    if len(file_list) != len(phases_list):
        raise ValueError("`files` and `phases` must have the same length.")

    
    print(f"File(s):{file_list} \nPhases: {phases_list}")

    # Function to append NaN rows for all stats per organ 
    def add_nan_row(file_path, phase, organ):
        row = dict(nan_template)
        row.update({
            "image": file_path,
            "contrast": phase,
            "organ": organ,
        })
        all_results.append(row)

    # Function to append NaN rows for organs per image 
    def add_nan_image(file_path, phase):
        for organ in organ_names:
            add_nan_row(file_path, phase, organ)


    print("Starting the extraction loop")

    for img_id, (path, phase) in enumerate(zip(file_list, phases_list)):
        file_path = PureWindowsPath(path).name
        
        print(f"Checking image {file_path}")

        try:
            images, segments = data_load(
                path,
                folder_path,
                channel_first=channel_first,
                organ_seg=organ_seg,
                organs=organs,
                by=by,
            )

        # Notify if image/mask loading fails and append NaN rows
        except Exception as exc:
            print(f"NaNs for {file_path}: failed to load image/mask ({exc})")
            add_nan_image(file_path, phase)
            continue
        
        # Append NaN row if image/mask list was empty
        if len(images) == 0 or len(segments) == 0:
            print(f"NaNs for {file_path}: missing image or mask data")
            add_nan_image(file_path, phase)
            continue

        img_vol, seg_vol = images[0], segments[0]
        
        if img_vol.ndim == 4:
            img_vol = img_vol[0] # remove image channel => [H, W, D]; C = 1 always

    
        if (
            img_vol.size == 0                     # empty image arrays
            or not np.any(np.isfinite(img_vol))   # non-finite values
            or np.count_nonzero(img_vol) == 0     # zero-intensity volumes
            or seg_vol.ndim < 4                   # incorrect mask shape, correct = [num_organs, H, W, D]
        ):
   
            print(f"NaNs for {file_path}: invalid image or mask")
            add_nan_image(file_path, phase)
            continue


    # ------------------------ CHECKS DONE ------------------------

        print(f"Checks done! Loading image {img_id}")

        image_sitk = sitk.GetImageFromArray(img_vol)

        for organ_idx in range(seg_vol.shape[0]):
            organ_name = organ_names[organ_idx]
            mask = seg_vol[organ_idx]

            # Fill in NaN values if the selected ROI is not present => empty mask (only 0s)
            if mask.sum() == 0: 
                print(f"Selected organ: {organ_name} is not present in the image")
                add_nan_row(file_path, phase, organ_name)
                continue
            
            print(f"Loading organ {organ_name}")

            try:
                res = extractor.execute(image_sitk, sitk.GetImageFromArray(mask.astype(np.uint8)))

            # Fill in NaN values if feature extraction fails
            except Exception as exc:
                print(f"NaNs for {file_path}, {organ_name}: feature extraction failed ({exc})")
                add_nan_row(file_path, phase, organ_name)
                continue

            # Add metadata
            res = {k: v for k, v in res.items() if not k.startswith("diagnostics")}
            res.update({
                "image": file_path,
                "contrast": phase,
                "organ": organ_name,
            })
            all_results.append(res)

        print(f"Processed image {img_id}") 


    df = pd.DataFrame(all_results)
    front_cols = ['image', 'contrast', 'organ']

    # Handle empty rows
    if df.empty:
        return pd.DataFrame(columns=front_cols)

    return df[front_cols + [c for c in df.columns if c not in front_cols]]

def append_df_to_csv(df, output_csv_path, wrote_header, processed_images, total_rows):
    processed_images += 1
    if df.empty:
        return wrote_header, processed_images, total_rows

    df.to_csv(
        output_csv_path,
        mode="w" if not wrote_header else "a",
        header=not wrote_header,
        index=False,
    )
    wrote_header = True
    total_rows += len(df)
    return wrote_header, processed_images, total_rows

def process_single_image(args):
    file_path, phase, folder_path = args
    return feature_extract(file_path, phase, folder_path)