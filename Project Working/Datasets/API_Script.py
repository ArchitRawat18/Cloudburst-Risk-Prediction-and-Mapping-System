import cdsapi
import os
from datetime import date, timedelta

c = cdsapi.Client()

# --- 1. PROJECT CONFIGURATION ---
START_YEAR = 2010
END_YEAR = 2025
MONSOON_MONTHS = ['06', '07', '08', '09']
# Himalayan Box: North, West, South, East
AREA_HIMALAYAS = [40, 65, 20, 100] 

# --- 2. PARAMETER LISTS ---
# Single Level Variables (Surface)
VARIABLES_SL = [
    'total_precipitation', 'surface_pressure', 'geopotential',
    '2m_temperature', '2m_dewpoint_temperature',
    'top_net_thermal_radiation',
    '10m_u_component_of_wind', '10m_v_component_of_wind',
    '100m_u_component_of_wind', '100m_v_component_of_wind'
]

# Pressure Level Variables (Upper Air)
VARIABLES_PL = [
    'relative_humidity', 'specific_humidity',
    'u_component_of_wind', 'v_component_of_wind'
]
LEVELS_PL = ['925', '850', '700', '500', '300', '200']

# --- 3. EXECUTION LOOP ---
for year in range(START_YEAR, END_YEAR + 1):
    # Create yearly folder structure
    year_folder = f'Himalaya_ERA5_Data/{year}'
    if not os.path.exists(year_folder):
        os.makedirs(year_folder)
    
    for month in MONSOON_MONTHS:
        # 10-day chunk logic: 1-10, 11-20, 21-end
        for chunk_start in [1, 11, 21]:
            # Calculate end day of chunk
            start_dt = date(year, int(month), chunk_start)
            if chunk_start == 21:
                # Get last day of the current month
                next_month = start_dt.replace(day=28) + timedelta(days=4)
                end_dt = next_month - timedelta(days=next_month.day)
            else:
                end_dt = start_dt + timedelta(days=9)
            
            day_list = [f'{d:02d}' for d in range(start_dt.day, end_dt.day + 1)]
            time_label = f"{year}_{month}_{chunk_start:02d}"

            # --- REQUEST A: SINGLE LEVELS ---
            sl_filename = f"{year_folder}/SL_{time_label}.nc"
            if not os.path.exists(sl_filename):
                print(f"Requesting SL: {year} month {month} days {day_list[0]}-{day_list[-1]}")
                c.retrieve('reanalysis-era5-single-levels', {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': VARIABLES_SL,
                    'year': str(year),
                    'month': month,
                    'day': day_list,
                    'time': [f'{h:02d}:00' for h in range(24)],
                    'area': AREA_HIMALAYAS,
                }, sl_filename)

            # --- REQUEST B: PRESSURE LEVELS ---
            pl_filename = f"{year_folder}/PL_{time_label}.nc"
            if not os.path.exists(pl_filename):
                print(f"Requesting PL: {year} month {month} days {day_list[0]}-{day_list[-1]}")
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

print("Batch download complete.")