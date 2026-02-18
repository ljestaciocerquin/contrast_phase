from monai.transforms import Compose, LoadImage, EnsureChannelFirst
from matplotlib.patches import Patch
import pandas as pd 
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from pathlib import PureWindowsPath
import os, sys
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
    Convert a single-label mask to a multi-channel mask [num_organs, H, W, D]

    organ_ids: list of int
               The integer labels corresponding to each organ
    seg_ct: np.ndarray 
        The single-label mask 

    returns channels: np.ndarray of shape [num_organs, H, W, D]
    """
    
    seg_ct = seg_ct[0]  # remove single-channel dim
    channels = np.zeros((len(organ_ids), *seg_ct.shape), dtype=np.uint8)
    for i, oid in enumerate(organ_ids):
        channels[i] = (seg_ct == oid).astype(np.uint8)
    return channels
    
def data_load(files, folder_path, channel_first = True, organ_seg = True):
    """
    Load the CT image files and organ segmentation files (optioonally)

    files: list of str
            List of all the file paths from which we want to load the images
    folder_path: str
            The folder path where all the CT images/organ segmentations are located
    channel_first: Boolean
            Ensure each loaded image has the channel first, i.e., is of shape [1, H, W, num_slices]
    organ_seg: Boolean 
            Specify whether you want to load organ segmentations 

    returns images (monai object), segments (monai object) 
    """
        
    images = [] 
    segments = []  
    loader = LoadImage(image_only=True)
    
    for file in files:
        file_dir = os.path.join(folder_path, PureWindowsPath(file).name)
        ct = loader(file_dir)
        if channel_first:
            ct = EnsureChannelFirst()(ct)

        ct = ct.numpy()
        images.append(ct)

    if organ_seg:
        organ_ids, _ = list_organs()
        organ_segments = [str(Path(f).with_name(Path(f).name.replace(".nii.gz", ".organs.nii.gz"))) for f in files]
        for seg in organ_segments:
            seg_dir = os.path.join(folder_path, PureWindowsPath(seg).name)
            seg_ct = loader(seg_dir)
            if channel_first:
                seg_ct = EnsureChannelFirst()(seg_ct)
            seg_ct = seg_ct.numpy()

            if organ_ids is not None:
                # Convert single-label segmentation -> multi-channel
                seg_ct = multi_channel(organ_ids, seg_ct)

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

def feature_extract(files, phases, folder_path):
    """
    Convert a single-label mask to a multi-channel mask [num_organs, H, W, D]

    organ_ids: list of int
               The integer labels corresponding to each organ
    seg_ct: np.ndarray 
        The single-label mask 

    returns channels: np.ndarray of shape [num_organs, H, W, D]
    """

    images, segments = data_load(files, folder_path)
    _, organ_names = list_organs()
    extractor = create_extractor()
    all_results = []

    print("starting the extraction loop")

    for img_id, (img_vol, seg_vol) in enumerate(zip(images, segments)):

        if img_vol.ndim == 4:
            img_vol = img_vol[0]

        for organ_idx in range(seg_vol.shape[0]):

            mask = seg_vol[organ_idx]

            if mask.sum() == 0: # skip empty masks
                continue

            image_sitk = sitk.GetImageFromArray(img_vol)
            mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
            
            res = extractor.execute(image_sitk, mask_sitk)

            # add metadata
            res['image'] = files[img_id]
            res['organ'] = organ_names[organ_idx]
            res['contrast'] = phases[img_id]
            res = {k: v for k, v in res.items() if not k.startswith('diagnostics')}
            all_results.append(res)
        
        print(f"Processed image {img_id}") 

    df = pd.DataFrame(all_results)
    front_cols = ['image', 'contrast', 'organ']
    df = df[front_cols + [c for c in df.columns if c not in front_cols]]

    return df

def save_results(df,path):
    """
    Save the dataframe to a given location
    """

    df.to_csv(path, index = False)