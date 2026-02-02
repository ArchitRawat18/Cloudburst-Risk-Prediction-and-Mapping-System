import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import plotly.express as px
import joblib
import tensorflow as tf
import os
import base64
from gtts import gTTS

# ================= ⚙️ CONFIGURATION =================
MODEL_DIR = r'F:\Major Project\Project Working\Modelling_i2'
DATA_DIR = r'F:\Major Project\Project Working\Datasets\Hardened_Data'

@st.cache_resource
def load_hybrid_models():
    rf = joblib.load(os.path.join(MODEL_DIR, "rf_spatial_engine.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    lstm = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_temporal_engine_i2.keras"))
    return rf, scaler, lstm

rf_model, scaler_obj, lstm_model = load_hybrid_models()
FEATURES = ['tp', 'z', 'q', 't', 'u', 'v', 'r', 'z_dip', 'r_spike', 'tp_trend']

# ================= 🎙️ LOCALIZED AUDIO ALERTS =================
def speak_alert(risk_score, location_name):
    if risk_score > 0.5:
        alert_text = f"Warning. High risk of {risk_score:.0%} detected at coordinates {location_name}."
        tts = gTTS(text=alert_text, lang='en')
        tts.save("local_alert.mp3")
        with open("local_alert.mp3", "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            st.markdown(f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)

# ================= 🛰️ DATA PROCESSING =================
def process_full_map(ds, hour_idx):
    # Feature Engineering
    ds = ds.assign(
        z_dip=ds['z'].diff('valid_time').fillna(0),
        r_spike=ds['r'].diff('valid_time').fillna(0),
        tp_trend=ds['tp'].rolling(valid_time=3).mean().fillna(0)
    )
    
    # RF Heatmap Baseline
    hour_ds = ds.isel(valid_time=hour_idx)
    df_step = hour_ds[FEATURES].to_dataframe().reset_index().dropna()
    scaled_rf = scaler_obj.transform(df_step[FEATURES])
    rf_probs = np.sum(rf_model.predict_proba(scaled_rf)[:, 1:], axis=1).reshape(ds.latitude.size, ds.longitude.size)
    
    return rf_probs, ds

# ================= 🖥️ UI LAYOUT =================
st.set_page_config(page_title="ACPS V3.1 Precision", layout="wide")
st.title("⛈️ Advanced Cloudburst Prediction System (V3.1)")

st.sidebar.header("Focus: 2024-2025 Events")
year = st.sidebar.selectbox("Year", ["2024", "2023"])
year_path = os.path.join(DATA_DIR, year)
selected_file = st.sidebar.selectbox("Event Log", [f for f in os.listdir(year_path) if f.endswith('.nc')])

if selected_file:
    ds_raw = xr.open_dataset(os.path.join(year_path, selected_file))
    if 'pressure_level' in ds_raw.dims: ds_raw = ds_raw.isel(pressure_level=0)
    
    time_idx = st.slider("Timeline (Observation Hour)", 24, len(ds_raw.valid_time)-1)
    
    with st.spinner("Generating Hybrid Heatmap..."):
        risk_map, ds_engineered = process_full_map(ds_raw, time_idx)

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("🗺️ Regional Risk Hotspots")
        fig = px.imshow(risk_map, x=ds_raw.longitude.values, y=ds_raw.latitude.values,
                        color_continuous_scale="Turbo", origin='lower')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Deep Coordinate Inference")
        c_lat, c_lon = st.columns(2)
        sel_lat = c_lat.selectbox("Latitude", np.round(ds_raw.latitude.values, 2))
        sel_lon = c_lon.selectbox("Longitude", np.round(ds_raw.longitude.values, 2))
        
        if st.button("Run Deep Inference"):
            lat_idx = np.abs(ds_raw.latitude.values - sel_lat).argmin()
            lon_idx = np.abs(ds_raw.longitude.values - sel_lon).argmin()
            
            # --- RF LOCAL (Spatial Snapshot) ---
            pixel_data = ds_engineered[FEATURES].isel(valid_time=time_idx, latitude=lat_idx, longitude=lon_idx)
            # FIX: Convert to DataFrame for scaler
            pixel_df = pd.DataFrame([pixel_data.to_pandas()])
            rf_local = np.sum(rf_model.predict_proba(scaler_obj.transform(pixel_df))[:, 1:], axis=1)[0]
            
            # --- LSTM LOCAL (Fixed Temporal Stagnation) ---
            # Extract raw 24h sequence for this specific coordinate
            seq_raw = ds_engineered[FEATURES].isel(latitude=lat_idx, longitude=lon_idx, 
                                                  valid_time=slice(time_idx-24, time_idx)).to_array().values
            # Reorder to (Time, Features)
            seq_raw = seq_raw.T 
            # Apply LOCAL MIN-MAX to highlight specific trend spikes
            seq_scaled = (seq_raw - seq_raw.min(axis=0)) / (seq_raw.max(axis=0) - seq_raw.min(axis=0) + 1e-7)
            lstm_local = lstm_model.predict(seq_scaled.reshape(1, 24, 10), verbose=0)[0][3]
            
            final_local = (rf_local * 0.7) + (lstm_local * 0.3)
            
            # --- METRICS & ALERTS ---
            m1, m2, m3 = st.columns(3)
            m1.metric("Spatial (RF)", f"{rf_local:.1%}")
            m2.metric("Temporal (LSTM)", f"{lstm_local:.1%}")
            st.metric("FINAL FUSED RISK", f"{final_local:.1%}")
            
            if final_local > 0.4:
                st.error(f"🚨 CRITICAL ALERT: Cloudburst probable at {sel_lat}, {sel_lon}")
                speak_alert(final_local, f"{sel_lat}, {sel_lon}")
            
            # --- VISUAL VERIFICATION ---
            # FIX: AttributeError resolved by using .to_pandas()
            st.write("#### Local Feature Signature")
            st.bar_chart(pixel_data.to_pandas())
            
            st.write("#### 24h Humidity Trend (Temporal Analysis)")
            trend = ds_engineered['r'].isel(latitude=lat_idx, longitude=lon_idx, 
                                            valid_time=slice(time_idx-24, time_idx)).values
            st.line_chart(trend)

    with st.expander("📝 Project Defense: Model Transparency"):
        st.write("""
        - **LSTM Variance**: By switching from global file scaling to local sequence scaling, the LSTM now 
          detects relative changes in the 24-hour lead-up for each specific coordinate.
        - **Cloudburst vs Rainfall**: The 'Temporal Trend' metric specifically looks for the sudden spikes 
          and pressure dips that distinguish a cloudburst from steady monsoon rain.
        """)