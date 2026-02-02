import cdsapi
import os
import xarray as xr
from datetime import date, timedelta

c = cdsapi.Client()

# --- CONFIGURATION ---
START_YEAR = 2010
END_YEAR = 2025
MONSOON_MONTHS = ['06', '07', '08', '09']
AREA_HIMALAYAS = [40, 65, 20, 100]
BASE_PATH = r'F:\Major Project\Project Working\Datasets\Himalaya_ERA5_Data'

VARIABLES_PL = [
    'temperature', 'relative_humidity', 'specific_humidity',
    'u_component_of_wind', 'v_component_of_wind', 'geopotential'
]
LEVELS_PL = ['925', '850', '700', '500', '300', '200']

for year in range(START_YEAR, END_YEAR + 1):
    year_folder = os.path.join(BASE_PATH, str(year))
    
    for month in MONSOON_MONTHS:
        for chunk_start in [1, 11, 21]:
            time_label = f"{year}_{month}_{chunk_start:02d}"
            pl_filename = os.path.join(year_folder, f"PL_{time_label}.nc")
            
            # --- SMART CHECK ---
            if os.path.exists(pl_filename):
                try:
                    with xr.open_dataset(pl_filename) as ds:
                        # If 't' (temperature) is already in the file, we skip it
                        if 't' in ds.data_vars:
                            print(f"Skipping {time_label}: Already updated with Temperature.")
                            continue
                except Exception:
                    print(f"File {time_label} is corrupted or unreadable. Re-downloading...")

            # --- DOWNLOAD LOGIC ---
            start_dt = date(year, int(month), chunk_start)
            if chunk_start == 21:
                next_month = start_dt.replace(day=28) + timedelta(days=4)
                end_dt = next_month - timedelta(days=next_month.day)
            else:
                end_dt = start_dt + timedelta(days=9)
            
            day_list = [f'{d:02d}' for d in range(start_dt.day, end_dt.day + 1)]
            
            print(f"Updating PL File: {year}-{month}-{chunk_start}")
            try:
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
            except Exception as e:
                print(f"Error updating {pl_filename}: {e}")

print("Update process finished.")