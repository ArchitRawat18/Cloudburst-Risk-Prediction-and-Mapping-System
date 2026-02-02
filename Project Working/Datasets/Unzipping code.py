import xarray as xr
import os
import zipfile
import shutil

base_path = r'F:\Major Project\Project Working\Datasets\Himalaya_ERA5_Data'

for year in range(2010, 2026):
    year_path = os.path.join(base_path, str(year))
    if not os.path.isdir(year_path):
        continue
    
    print(f"--- Processing Year: {year} ---")
    
    # List all SL files in the folder
    all_files = [f for f in os.listdir(year_path) if f.startswith('SL_') and f.endswith('.nc')]
    
    for filename in all_files:
        full_path = os.path.join(year_path, filename)
        
        # 1. Check if the file is a ZIP archive
        if zipfile.is_zipfile(full_path):
            try:
                # 2. Create a temporary 'sandbox' folder for THIS specific ZIP
                temp_folder = os.path.join(year_path, f"temp_{filename}")
                os.makedirs(temp_folder, exist_ok=True)
                
                with zipfile.ZipFile(full_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_folder)
                
                # 3. Process the extracted files inside the sandbox
                extracted_files = os.listdir(temp_folder)
                for ext_file in extracted_files:
                    ext_path = os.path.join(temp_folder, ext_file)
                    
                    # Open file to read the internal timestamp
                    with xr.open_dataset(ext_path) as ds:
                        # Extract the start date (e.g., 2010-06-01)
                        start_time = str(ds.valid_time.values[0])[:10].replace('-', '_')
                        # Identify if it is accum or inst
                        suffix = 'accum' if 'accum' in ext_file else 'inst'
                        new_name = f"SL_{start_time}_{suffix}.nc"
                    
                    # 4. Move the file out of the sandbox to the main folder
                    final_destination = os.path.join(year_path, new_name)
                    shutil.move(ext_path, final_destination)
                    print(f" Extracted & Renamed: {new_name}")
                
                # 5. Cleanup: Remove the sandbox and the original ZIP
                shutil.rmtree(temp_folder)
                os.remove(full_path)
                
            except Exception as e:
                print(f" Error processing {filename}: {e}")
                # If error occurs, try to clean up the temp folder to stay tidy
                if os.path.exists(temp_folder):
                    shutil.rmtree(temp_folder)
        else:
            # If it's a normal .nc file (already unzipped), just skip it
            print(f" Skipping {filename} (Already a standard NetCDF)")

print("\nAll files unzipped and uniquely renamed successfully.")