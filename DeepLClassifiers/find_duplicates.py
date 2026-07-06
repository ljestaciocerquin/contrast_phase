from pathlib import Path
from typing import Iterable, Dict, List

def find_duplicate_filenames(folders: Iterable[str]) -> Dict[str, List[str]]:
    # This is just to check if there are patients across all the folders: train, val, test
    # Ideally your data should be split at patient-level, not image-level
    file_locations: Dict[str, List[str]] = {}

    for folder in folders:
        path = Path(folder)

        if not path.is_dir():
            raise ValueError(f'Invalid directory: {folder}')
        
        for item in path.iterdir():
            if item.is_file():
                name =item.name

                if name not in file_locations:
                    file_locations[name] = [str(path)]
                else:
                    file_locations[name].append(str(path))
    duplicates = {
        name: locations 
        for name, locations in file_locations.items()
        if len(locations) > 1
    }

    return duplicates

train = '/mnt/rhea/data_private/IRBd23-231/GEPNETs/contrast_phase/diffusion_data/train'
test  = '/mnt/rhea/data_private/IRBd23-231/GEPNETs/contrast_phase/diffusion_data/test'
val   = '/mnt/rhea/data_private/IRBd23-231/GEPNETs/contrast_phase/diffusion_data/val'
folders  = [train, test, val]
duplicates = find_duplicate_filenames(folders)

if duplicates:
    print("Duplicate files found across folders:")
    for name, locations in duplicates.items():
        print(f"{name} -> {locations}")
else:
    print("No duplicates found.")
