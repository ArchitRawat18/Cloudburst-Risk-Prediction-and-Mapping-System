import xarray as xr
import os

# Paths
target_dir = r'F:\Major Project\Project Working\Datasets\Himalaya_ERA5_Data'

# Define the Western Himalayan Bounding Box
lat_bounds = slice(37.5, 28.5) # ERA5 lats are usually North to South (high to low)
lon_bounds = slice(72.0, 81.5)

def crop_dataset(directory):
    total_files = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nc'):
                file_path = os.path.join(root, file)
                try:
                    # Use chunks to handle memory efficiently
                    with xr.open_dataset(file_path, chunks={'time': 100}) as ds:
                        # Check if coordinates exist in the file
                        if 'latitude' in ds.coords and 'longitude' in ds.coords:
                            # Slice the data
                            cropped_ds = ds.sel(latitude=lat_bounds, longitude=lon_bounds)
                            
                            # Use a temporary file to avoid corruption during write
                            temp_path = file_path + ".temp"
                            cropped_ds.to_netcdf(temp_path)
                            
                    # Replace original with cropped
                    os.replace(temp_path, file_path)
                    print(f"Cropped: {file}")
                    total_files += 1
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    print(f"\nProcessing Complete. {total_files} files cropped to Western Himalayan bounds.")

# Run the cropping
crop_dataset(target_dir)