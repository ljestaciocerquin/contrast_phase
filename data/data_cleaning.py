import pandas as pd
import numpy as np
from pathlib import PureWindowsPath
import os 
from pathlib import Path

def flag_existing(data):
    exists_mask = list(zip(data['SubjectKeyRadiology'], data['ExamDate']))

def fix_columns(processed, data):
    cols = processed.columns.tolist()
    cols.extend(['SliceThickness'])
    cols.insert(0, 'Original File')
    cols.insert(1, 'Destination Folder')
    cols.insert(2, 'New File')
    cols = [col for col in cols if col not in ['SegmentationBatch', 'NiiFile', 'SegNiiFile']]

    data = data.rename(columns={'no_lesions': 'is_lesionfree'})
    data = data[cols]
    
    return data


def annot_mapping(data, verbose = 0):
    contrast_mapping = {"1": "Arterial", "2": "Portal", "NO_CONTRAST": "Non-contrast", "3": "Late Phase", "0": "0"}  # contrast = 0 (forgot to select) or NA (no liver - skip, ignore from dataset)
    time_mapping = {1.0: "Too Early", 2.0: "Just Right", 3.0: "Too Late", 0.0: 0.0}                                  # phase_timing = NA ??? (skip, ignore from dataset)
    liver_mapping = {1.0: "Yes", 2.0: "Partially", 0.0: "0"}                                                         # is_liver_imaged = "0" or NA ???
    lesion_mapping = {True: "Yes", None: "No"}

    data['contrast'] = data['contrast'].map(contrast_mapping)
    data['phase_timing'] = data['phase_timing'].map(time_mapping)
    data['is_liver_imaged'] = data['is_liver_imaged'].map(liver_mapping)
    data.loc[data['file'].notna(), 'is_lesionfree'] = data.loc[data['file'].notna(), 'is_lesionfree'].map(lesion_mapping).fillna("No")

    # If contrast == 0 and ProtocolName contains -c => contrast = NC
    mask_nc = (
        # data['contrast'].eq("0") &
        data['ProtocolName'].str.contains(r'-c(?=\W|$)', case=False, na=False) # -C before anything that is not a letter or number
    )

    data.loc[mask_nc, 'contrast'] = 'Non-contrast'

    # If contrast == NC => phase_timing = NC
    mask = data['contrast'].eq('Non-contrast')
    data.loc[mask, 'phase_timing'] = 'Non-contrast'


    # If is_liver_imaged is 0 or NA and contrast is not NA and BodyPartExamined/ ProtocolName contains lever or abdomen => is_liver_imaged = Yes
    mask_liver = (
        (data['is_liver_imaged'].eq("0") | data['is_liver_imaged'].isna()) &
        (data['BodyPartExamined'].str.lower().str.match(r'\blever\b|\babdomen\b', case=False,na=False) |
        data['ProtocolName'].str.lower().str.match(r'\blever\b|\babdomen\b', case=False,na=False))
    )

    data.loc[mask_liver, 'is_liver_imaged'] = 'Yes'

    if verbose == 1:
        print(data.contrast.value_counts(), f"Null values: {int(data.contrast.isna().sum())}",
                data.is_liver_imaged.value_counts(), f"Null values: {int(data.is_liver_imaged.isna().sum())}",
                data.phase_timing.value_counts(), f"Null values: {int(data.phase_timing.isna().sum())}",
                data.is_lesionfree.value_counts(), f"Null values: {int(data.is_lesionfree.isna().sum())}")

    return data


def exclude_cor_sag(data, verbose = 0):
    mask_cor_sag = data['ProtocolName'].str.contains(
        r'(?i)(^|[^a-zA-Z])(cor|coronal|coronaal|sag|sagitaal)(?=$|[^a-zA-Z])',
        na=False
    )
    data = data[~mask_cor_sag]

    if verbose ==1:
        print("ProtocolName values (True values excluded):\n",mask_cor_sag.value_counts())
        print(data.contrast.value_counts(), f"Null values: {int(data.contrast.isna().sum())}",
            data.is_liver_imaged.value_counts(), f"Null values: {int(data.is_liver_imaged.isna().sum())}",
            data.phase_timing.value_counts(), f"Null values: {int(data.phase_timing.isna().sum())}",
            data.is_lesionfree.value_counts(), f"Null values: {int(data.is_lesionfree.isna().sum())}")

    return data

def check_zeros(data):
    """
    Check the samples that have 0.0 values for contrast/phase_timing/is_liver_imaged
    If not empty, save them for validation
    """
    zeros = data[(data.contrast == "0") 
                                        | (data.is_liver_imaged == "0")
                                        | (data.phase_timing == 0.0)
              ]

    zeros.to_csv("/projects/net_contrast_classification/contrast_phase/data/kalina_checks/to_check/unknown_contrast_timing_or_liver.csv", index=False)
    return zeros

# def reformat_time(data, time_col):
#     data[str(time_col, "sec")] = pd.to_timedelta(data[time_col]).dt.total_seconds()
#     return data

# def reformat_date(data):
#     data["ExamDate"] = pd.to_datetime(data["ExamDate"])
#     return data

def reformat_missing_bodypart(data):

    mask = data['BodyPartExamined'].isna()
    # extract first word from ProtocolName
    first_word = (
        data.loc[mask, 'ProtocolName']
        .astype(str)
        .str.strip()
        .str.split()
        .str[0]
        .str.upper()
    )

    # allowed mappings
    mapping = {
        'PANCREAS': 'PANCREAS',
        'SOFT': 'SOFT TISSUE',
        'ABDOMEN': 'ABDOMEN',
        'ABDOMEN+': 'ABDOMEN',
        'Abdomen': 'ABDOMEN',
        'Abd.': 'ABDOMEN',
        'Lever': 'LEVER',
        'LEVER': 'LEVER',
        'lever': 'LEVER',
        'BODY': 'BODY',
        'Body': 'BODY',
        'Mediastinum': 'MEDIASTINUM',
        'MEDIASTINUM': 'MEDIASTINUM'
    }

    # fill only if first word is in mapping
    data.loc[mask, 'BodyPartExamined'] = first_word.map(mapping)

    print("Number of missing BodyPartExamined values:", int(data['BodyPartExamined'].isna().sum()))

    return data

def get_slice_thickness(data, verbose = 0):
    # Replace commas with dots in numbers
    data['ProtocolName'] = data['ProtocolName'].str.replace(r'(\d+),(\d+)', r'\1.\2', regex=True)

    mask = data['SliceThickness'].isna()

    data.loc[mask, 'SliceThickness'] = (
        data.loc[mask, 'ProtocolName']
            .str.extract(r'(\d+\.\d+|\d+(?=\s*[mM]{2}))')[0]
            .astype(float)
    )
    
    if verbose == 1:
        print(f"\nNumber of remaining missing values in SliceThickness after extraction: {data.SliceThickness.isna().sum()}")
        print(data.SliceThickness.value_counts())
    
    return data

def map_dicom_phase(protocol_name):
    protocol_name = str(protocol_name).lower()  # normalize
    if any(k in protocol_name for k in [' a ', ' art', ' art.', 'laat-art', 'arterieel', 'arterial']):
        return 'Arterial'
    elif any(k in protocol_name for k in ['p ', ' pv ',' v ', 'portaal', 'venous', 'veneus', 'ven', 'port', ' vv ']):
        if any(k in protocol_name for k in ['laat veneus']): #
            return 'Late Venous'
        return 'Portal'
    elif any(k in protocol_name for k in ['late fase', 'delayed phase', 'min ', 'min. ']):
        return 'Late Phase'
    elif any(k in protocol_name for k in [' -c ']):
        return 'Non-contrast'
    elif any(k in protocol_name for k in ['uitscheiding']):
        return 'Excretory'
    else:
        return np.nan  # unknown / other

def check_duplicates(data):
    duplicated = data[data[['SubjectKeyRadiology', 'ExamDate', 'file', 'ProtocolName']].duplicated(keep=False)].sort_values(['SubjectKeyRadiology', 'Original File'])

    if len(duplicated) == 0:
        print("No duplicates found!")
    else:
        print(f"Found {len(duplicated)} duplicates")
        print(duplicated)

def data_stats(data):
    print('Number of patients:', data['SubjectKeyRadiology'].nunique(), 
      '\nNumber of files:', data['Original File'].nunique(), 
      '\nNumber of patients with both arterial and portal:', data[data.contrast.isin(['Arterial', 'Portal'])]
                                                                                .groupby('SubjectKeyRadiology')['contrast']
                                                                                .nunique()
                                                                                .eq(2)
                                                                                .sum()
                                                                                )

def different_dicom_phase(data):
    diff_dicom = data[(data["DICOM_phase"] != data["contrast"]) 
                  & (data["DICOM_phase"].notna()
                  & data.contrast.notna()) 
                  ][["SubjectKeyRadiology","ExamDate", "ProtocolName", "contrast", "phase_timing", "DICOM_phase", "is_liver_imaged", 'comment', 'other']]
    # diff_dicom.to_csv("diff_dicom.csv", index=False)

    print("Entries with contrast label different than the DICOM label", len(diff_dicom))
    print("Distribution of the DICOM inconsistencies:\n",diff_dicom.contrast.value_counts())
    
def remove_late_acquisitions(data):
    data_cleared = data[~((data["contrast"] == "Late Phase") |
                    (data["DICOM_phase"].isin(["Excretory", "Late Phase"])))]

    return data_cleared


# def add_nii_paths(data):
#     out_folder = Path(r"Z:\_archived")

#     data.insert(0, "NiiFile", Path(out_folder) / data['Destination Folder']/ data['New File'])
#     out_folder = Path(r"Z:\_archived")

#     # Function to select the correct seg file path
#     def seg_path(row):
#         folder = out_folder / row['Destination Folder']
#         base_name = row['New File'].replace(".nii.gz", "")
#         seg2 = folder / f"{base_name}.seg_2.nii.gz"
#         seg1 = folder / f"{base_name}.seg.nii.gz"
#         if seg2.exists():
#             return seg2
#         elif seg1.exists():
#             return seg1
#         else:
#             return pd.NA

#     # Create column efficiently
#     data.insert(1, "SegNiiFile", [seg_path(row) for _, row in data.iterrows()])
#     return data

def add_matchkey(data, col, fallback_label='unknown'):
    file_part = data['file'].astype('string').str.split('_').str[1]
    file_part = file_part.fillna(fallback_label) + '.nii.gz'   # pick your fallback label

    data[col] = (
        data['SubjectKeyRadiology'].astype('string')
        + "_"
        + data['Original File'].astype('string').apply(lambda x: PureWindowsPath(x).parts[3].split(" ")[0])
        + "_"
        + data['Destination Folder'].astype('string').str.split('-').str[-1]
        + "_"
        + file_part
    )
    return data




def main():

    # ==============================================================================
    # Load & Merge
    # ==============================================================================

    selected = pd.read_csv('/projects/net_contrast_classification/contrast_phase/data/original_data/selected_scans.csv')
    processed = pd.read_csv('/projects/net_contrast_classification/contrast_phase/data/original_data/dataset-registration-artinet.csv')
    annot = pd.read_csv('/projects/net_contrast_classification/contrast_phase/data/original_data/annotations.csv', )
    dicoms = pd.read_csv('/projects/net_contrast_classification/contrast_phase/data/original_data/dicom_metadata.csv')

    selected['SubjectKeyRadiology'] = selected['Original File'].apply(lambda x: PureWindowsPath(x).parts[2])
    selected['ExamDate'] = selected['Original File'].apply(lambda x: pd.to_datetime(PureWindowsPath(x).parts[3].split(' ')[0]))

    selected_dicoms = selected.merge(dicoms, on=['Original File'], how='left')
    data = selected_dicoms.merge(annot, left_on=['Destination Folder','New File'], right_on=['folder', 'file'], how='right')
    data = fix_columns(processed, data)

    save_root = "/projects/net_contrast_classification/contrast_phase/data/original_data/full_data.csv"
    data.to_csv(save_root, index = False)

    # ==============================================================================
    # Mapping
    # ==============================================================================

    data = reformat_missing_bodypart(data)
    data = annot_mapping(data, verbose = 1)
    data = exclude_cor_sag(data, verbose = 1)
    check_zeros(data)

    data['AcquisitionTime_sec'] = pd.to_timedelta(data['AcquisitionTime']).dt.total_seconds()
    data["ExamDate"] = pd.to_datetime(data["ExamDate"])
    data = get_slice_thickness(data)

    data['DICOM_phase'] = data['ProtocolName'].apply(map_dicom_phase)


    # ==============================================================================
    # Check for duplicates
    # ==============================================================================
    check_duplicates(data)

    # print data stats
    data_stats(data)

    # check DICOM inconsistencies
    different_dicom_phase(data)

    # ==============================================================================
    # Create MatchKey
    # ==============================================================================

    data = add_matchkey(data, 'MatchKey')
    # ==============================================================================
    # Clean late acquisituons & Save
    # ==============================================================================

    data_cleared = remove_late_acquisitions(data)
    print("Length of data_cleared:", len(data_cleared))
    print(data_cleared.contrast.value_counts())

    save_root_cleaned = Path("/projects/net_contrast_classification/contrast_phase/data/cleaned_data")
    save_root_cleaned.mkdir(parents=True, exist_ok=True)     # ensure directory exists
    print(save_root_cleaned.exists())
    print(save_root_cleaned.resolve())
    output_file = save_root_cleaned / "cleaned_data_raw.csv"     
    data_cleared.to_csv(output_file, index=False)            # save file

    # ==============================================================================
    # Exists on server flag
    # ==============================================================================

    data_with_flag = pd.read_csv("/projects/net_contrast_classification/contrast_phase/data/cleaned_data/cleaned_data_server_flag.csv")
    final_data = data_cleared.merge(
                        data_with_flag[["MatchKey", "exist_on_server"]],
                        on=["MatchKey"],
                        how="left",
                        validate="many_to_one"
                        )
    print(len(final_data[final_data.SubjectKeyRadiology.notna()]), len(final_data[final_data.exist_on_server.notna()]))
    final_data.to_csv(save_root_cleaned / "cleaned_data.csv" , index=False)            # save file

if __name__ == "__main__":
    main()
