import cdsapi

c = cdsapi.Client()

# Path to the missing file
target_path = r'F:\Major Project\Project Working\Datasets\Himalaya_ERA5_Data\2025\SL_2025_06_11.nc'

print("Requesting missing chunk: 2025-06-11 to 2025-06-20")

c.retrieve('reanalysis-era5-single-levels', {
    'product_type': 'reanalysis',
    'format': 'netcdf',
    'variable': [
        'total_precipitation', 'surface_pressure', 'geopotential',
        '2m_temperature', '2m_dewpoint_temperature',
        'top_net_thermal_radiation', '10m_u_component_of_wind', 
        '10m_v_component_of_wind', '100m_u_component_of_wind', 
        '100m_v_component_of_wind'
    ],
    'year': '2025',
    'month': '06',
    'day': [str(d).zfill(2) for d in range(11, 21)], # Days 11 through 20
    'time': [f'{h:02d}:00' for h in range(24)],
    'area': [40, 65, 20, 100], # Your Himalayan Box
}, target_path)

print("Download complete.")