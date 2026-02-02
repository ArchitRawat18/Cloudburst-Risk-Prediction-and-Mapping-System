import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
import glob
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Himalayan Cloudburst Watch", layout="wide")
st.title("🏔️ Himalayan Cloudburst Early Warning System")

@st.cache_resource
def get_model():
    return load_model(r'F:\Major Project\Project Working\Modelling\cloudburst_final_v2.keras')

model = get_model()
DATA_DIR = r'F:\Major Project\Project Working\Datasets\Hardened_Data'

# --- 1. DATA STITCHING LOGIC ---
def get_full_history(target_datetime):
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.nc"), recursive=True))
    main_ds = None
    file_path = ""
    for f in all_files:
        temp_ds = xr.open_dataset(f)
        if target_datetime in pd.to_datetime(temp_ds.valid_time.values):
            main_ds = temp_ds
            file_path = f
            break
    if main_ds is None: return None, None
    times = pd.to_datetime(main_ds.valid_time.values)
    idx = np.where(times == target_datetime)[0][0]
    
    if idx < 24: # Pull from previous file if at the start of a month
        prev_idx = all_files.index(file_path) - 1
        if prev_idx >= 0:
            prev_ds = xr.open_dataset(all_files[prev_idx])
            combined_ds = xr.concat([prev_ds, main_ds], dim='valid_time')
            new_idx = np.where(pd.to_datetime(combined_ds.valid_time.values) == target_datetime)[0][0]
            return combined_ds, new_idx
    return main_ds, idx

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("Search Parameters")
    d = st.date_input("Select Date", value=pd.to_datetime("2024-07-31"))
    t = st.selectbox("Select Time", options=[f"{h:02d}:30" for h in range(24)], index=21)
    target_dt = pd.to_datetime(f"{d} {t}")

# --- 3. UPDATED PROCESSING ---
ds, hour_idx = get_full_history(target_dt)

if ds is not None:
    if 'pressure_level' in ds.dims: ds = ds.isel(pressure_level=0)
    
    # LSTM Sequence Prep
    features = ['tp', 'z', 'q', 't', 'u', 'v', 'r']
    data_vals = ds[features].to_array().transpose('valid_time','latitude','longitude','variable').values
    data_norm = (data_vals - data_vals.min()) / (data_vals.max() - data_vals.min() + 1e-7)
    
    lat_vals, lon_vals = ds.latitude.values, ds.longitude.values
    all_pixels = []
    for lat in range(len(lat_vals)):
        for lon in range(len(lon_vals)):
            all_pixels.append(data_norm[hour_idx-24:hour_idx, lat, lon, :])
    
    # 4. FIXED CALIBRATION SCALING
    preds = model.predict(np.array(all_pixels), batch_size=256, verbose=0)
    prob_map = preds[:, 3].reshape(len(lat_vals), len(lon_vals))
    
    # FIXED RANGE: Map model sensitivity (0.021 to 0.0235) to 0-100 Risk
    LOWER_LIMIT, UPPER_LIMIT = 0.021, 0.0235
    risk_score = ((prob_map - LOWER_LIMIT) / (UPPER_LIMIT - LOWER_LIMIT)) * 100
    risk_score = np.clip(risk_score, 0, 100) # Prevents negative/overflow values

    # --- 5. ENHANCED MAPS ---
    lons_grid, lats_grid = np.meshgrid(lon_vals, lat_vals) # Correct order for plotting
    df_risk = pd.DataFrame({
        'lat': lats_grid.flatten(), 
        'lon': lons_grid.flatten(), 
        'risk': risk_score.flatten()
    })

    st.subheader(f"🔥 Predicted Cloudburst Risk Intensity (0-100 Scale)")
    # Radius=30 creates the smooth 'AQI' color transition
    fig_risk = px.density_mapbox(df_risk, lat='lat', lon='lon', z='risk', radius=30, 
                                 center=dict(lat=32, lon=77), zoom=5,
                                 mapbox_style="open-street-map", color_continuous_scale="YlOrRd",
                                 range_color=[0, 100])
    fig_risk.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("---")
    
    # 6. GROUND TRUTH (SCROLLABLE BELOW)
    st.subheader("📍 Historical Ground Truth (Labels from ERA5)")
    actual_labels = ds.target.values[hour_idx]
    df_actual = pd.DataFrame({
        'lat': lats_grid.flatten(), 
        'lon': lons_grid.flatten(), 
        'label': actual_labels.flatten()
    })
    
    fig_actual = px.density_mapbox(df_actual, lat='lat', lon='lon', z='label', radius=20,
                                  center=dict(lat=32, lon=77), zoom=5,
                                  mapbox_style="open-street-map", color_continuous_scale="Blues",
                                  range_color=[0, 3])
    fig_actual.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_actual, use_container_width=True)
else:
    st.error("No data available for this date.")
