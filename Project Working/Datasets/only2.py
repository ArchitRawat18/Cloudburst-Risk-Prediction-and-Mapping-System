import cdsapi
import os
from datetime import date, timedelta

c = cdsapi.Client()

# --- CONFIGURATION ---
AREA_HIMALAYAS = [40, 65, 20, 100]
BASE_PATH = r'F:\Major Project\Project Working\Datasets\Himalaya_ERA5_Data'

VARIABLES_PL = [
    'temperature', 'relative_humidity', 'specific_humidity',
    'u_component_of_wind', 'v_component_of_wind', 'geopotential'
]
LEVELS_PL = ['925', '850', '700', '500', '300', '200']

# Define only the specific files you want to fix
# Format: (year, month, chunk_start)
TARGET_FILES = [
    (2012, '07', 1),
    (2025, '07', 1)
]

for year, month, chunk_start in TARGET_FILES:
    year_folder = os.path.join(BASE_PATH, str(year))
    
    # Ensure folder exists
    if not os.path.exists(year_folder):
        os.makedirs(year_folder)
        
    time_label = f"{year}_{month}_{chunk_start:02d}"
    pl_filename = os.path.join(year_folder, f"PL_{time_label}.nc")

    # --- DATE CALCULATION ---
    # We force the download for days 01 to 10 for these specific files
    start_dt = date(year, int(month), chunk_start)
    end_dt = start_dt + timedelta(days=9)
    day_list = [f'{d:02d}' for d in range(start_dt.day, end_dt.day + 1)]

    print(f"--- Redownloading: {pl_filename} ---")
    
    try:
        # Note: If the file exists, this will overwrite it with the full data
        c.retrieve('reanalysis-era5-pressure-levels', {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': VARIABLES_PL,
            'pressure_level': LEVELS_PL,
            'year': str(year),
            'month': month,
            'day': day_list,
            'time': [f'{h:02d}:00' for h in range(24)],
            'area': AREA_HIMALAYAS,
        }, pl_filename)
        print(f"Successfully updated {time_label}")
        
    except Exception as e:
        print(f"Error updating {pl_filename}: {e}")

print("\nSpecific update process finished.")