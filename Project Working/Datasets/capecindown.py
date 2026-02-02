import cdsapi
import os
from datetime import date, timedelta

c = cdsapi.Client()

# --- CONFIGURATION (Identical to your main script) ---
START_YEAR = 2010
END_YEAR = 2025
MONSOON_MONTHS = ['06', '07', '08', '09']
AREA_HIMALAYAS = [40, 65, 20, 100] 

for year in range(START_YEAR, END_YEAR + 1):
    year_folder = f'Himalaya_ERA5_Data/{year}'
    if not os.path.exists(year_folder):
        os.makedirs(year_folder)
    
    for month in MONSOON_MONTHS:
        for chunk_start in [1, 11, 21]:
            # Date Logic
            start_dt = date(year, int(month), chunk_start)
            if chunk_start == 21:
                next_month = start_dt.replace(day=28) + timedelta(days=4)
                end_dt = next_month - timedelta(days=next_month.day)
            else:
                end_dt = start_dt + timedelta(days=9)
            
            day_list = [f'{d:02d}' for d in range(start_dt.day, end_dt.day + 1)]
            time_label = f"{year}_{month}_{chunk_start:02d}"
            
            # Target Filename
            cape_filename = f"{year_folder}/SL_{time_label}_cape.nc"
            
            if not os.path.exists(cape_filename):
                print(f"Requesting CAPE/CIN: {year}-{month}-{chunk_start}")
                c.retrieve('reanalysis-era5-single-levels', {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': [
                        'convective_available_potential_energy',
                        'convective_inhibition'
                    ],
                    'year': str(year),
                    'month': month,
                    'day': day_list,
                    'time': [f'{h:02d}:00' for h in range(24)],
                    'area': AREA_HIMALAYAS,
                }, cape_filename)

print("CAPE and CIN download complete.")