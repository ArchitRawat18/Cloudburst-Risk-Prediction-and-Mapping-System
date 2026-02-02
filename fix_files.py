import zipfile
import os

def fix_zip_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".nc"):
                path = os.path.join(root, file)
                # Check if it's actually a zip
                if zipfile.is_zipfile(path):
                    print(f"Unzipping {path}...")
                    with zipfile.ZipFile(path, 'r') as zip_ref:
                        zip_ref.extractall(root)
                    os.remove(path) # Remove the zip-disguised-as-nc

fix_zip_files(r'F:\Major Project\Project Working\Datasets\Himalaya_ERA5_Data')