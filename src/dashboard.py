import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Load models
class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_size):
        super(AnomalyAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, input_size)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

@st.cache_resource
def load_models():
    with open(r'C:\Users\sanga\orbit-shield-ml\models\isolation_forest.pkl', 'rb') as f:
        iso = pickle.load(f)
    with open(r'C:\Users\sanga\orbit-shield-ml\models\scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    ae = AnomalyAutoencoder(14)
    ae.load_state_dict(torch.load(r'C:\Users\sanga\orbit-shield-ml\models\autoencoder.pth'))
    ae.eval()
    return iso, scaler, ae

iso_model, scaler, model_ae = load_models()

# Load data
columns = ['engine_id','time_cycle','op_setting_1','op_setting_2','op_setting_3',
           's1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12',
           's13','s14','s15','s16','s17','s18','s19','s20','s21']
df = pd.read_csv(r'C:\Users\sanga\orbit-shield-ml\data\train_FD001.txt', sep=r'\s+', header=None, engine='python')
df.columns = columns

useful_sensors = ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']

# Dashboard UI
st.title("Orbit Shield ML")
st.subheader("Real-time Rocket Telemetry Anomaly Detection")
st.markdown("Inspired by ISRO PSLV-C61 and PSLV-C62 failures")

# Engine selector
engine_id = st.selectbox("Select Engine ID", sorted(df['engine_id'].unique()))
engine_data = df[df['engine_id'] == engine_id]

# Sensor plot
sensor = st.selectbox("Select Sensor", useful_sensors)
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(engine_data['time_cycle'], engine_data[sensor], color='steelblue')
ax.set_title(f"Engine {engine_id} — Sensor {sensor}")
ax.set_xlabel("Time cycle")
st.pyplot(fig)

# Run anomaly detection
X = engine_data[useful_sensors]
X_scaled = scaler.transform(X)
X_tensor = torch.FloatTensor(X_scaled)

with torch.no_grad():
    reconstructed = model_ae(X_tensor)
errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
threshold = 0.187307

ae_pred = (errors > threshold).astype(int)
iso_pred = (iso_model.predict(X_scaled) == -1).astype(int)
ensemble = ((ae_pred == 1) & (iso_pred == 1)).astype(int)

anomaly_count = ensemble.sum()
total = len(ensemble)

# Status
st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("Total Readings", total)
col2.metric("Anomalies Detected", anomaly_count)
col3.metric("Anomaly %", f"{round(anomaly_count/total*100, 1)}%")

if anomaly_count > 0:
    st.error("ALERT: Anomalies detected in this engine's telemetry.")
else:
    st.success("All readings normal.")

# Anomaly timeline
fig2, ax2 = plt.subplots(figsize=(10, 2))
ax2.plot(engine_data['time_cycle'].values, errors, color='steelblue', label='Reconstruction error')
ax2.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
ax2.set_title("Anomaly Score over Time")
ax2.legend()
st.pyplot(fig2)
