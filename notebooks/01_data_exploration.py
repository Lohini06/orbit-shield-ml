import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import pickle
import shap

# ── 1. LOAD DATA ──────────────────────────────────────────────
columns = ['engine_id', 'time_cycle',
           'op_setting_1', 'op_setting_2', 'op_setting_3',
           's1','s2','s3','s4','s5','s6','s7','s8','s9',
           's10','s11','s12','s13','s14','s15','s16',
           's17','s18','s19','s20','s21']

df = pd.read_csv(r'C:\Users\sanga\orbit-shield-ml\data\train_FD001.txt',
                 sep=r'\s+', header=None, engine='python')
df.columns = columns
print("Data loaded. Shape:", df.shape)

# ── 2. ADD REMAINING USEFUL LIFE (RUL) ────────────────────────
rul = []
for engine_id in df['engine_id'].unique():
    engine_data = df[df['engine_id'] == engine_id]
    max_cycle = engine_data['time_cycle'].max()
    rul_values = max_cycle - engine_data['time_cycle']
    rul.extend(rul_values.tolist())
df['RUL'] = rul
print("RUL added.")

# ── 3. LABEL ANOMALIES ────────────────────────────────────────
df['anomaly'] = (df['RUL'] <= 30).astype(int)
print("Normal readings:", (df['anomaly'] == 0).sum())
print("Anomaly readings:", (df['anomaly'] == 1).sum())

# ── 4. FEATURE SELECTION ──────────────────────────────────────
useful_sensors = ['s2','s3','s4','s7','s8','s9','s11',
                  's12','s13','s14','s15','s17','s20','s21']
X = df[useful_sensors]
y = df['anomaly']
print("Features shape:", X.shape)

# ── 5. SCALING ────────────────────────────────────────────────
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("Scaling done. Min:", X_scaled.min(), "Max:", X_scaled.max())

# ── 6. TRAIN/TEST SPLIT ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
print("Train samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# ── 7. ISOLATION FOREST ───────────────────────────────────────
X_train_normal = X_train[y_train == 0]
iso_model = IsolationForest(contamination=0.05, random_state=42)
iso_model.fit(X_train_normal)
y_pred_iso = (iso_model.predict(X_test) == -1).astype(int)
print("\nIsolation Forest Results:")
print(classification_report(y_test, y_pred_iso))

# ── 8. AUTOENCODER ────────────────────────────────────────────
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

X_train_normal_tensor = torch.FloatTensor(X_train[y_train == 0])
X_test_tensor = torch.FloatTensor(X_test)

model_ae = AnomalyAutoencoder(input_size=14)
optimizer = torch.optim.Adam(model_ae.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("\nTraining Autoencoder...")
for epoch in range(50):
    model_ae.train()
    optimizer.zero_grad()
    output = model_ae(X_train_normal_tensor)
    loss = criterion(output, X_train_normal_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/50 — Loss: {loss.item():.6f}")

# ── 9. EVALUATE ENSEMBLE ──────────────────────────────────────
model_ae.eval()
with torch.no_grad():
    reconstructed = model_ae(X_test_tensor)
errors = torch.mean((X_test_tensor - reconstructed) ** 2, dim=1).numpy()
threshold = np.percentile(errors, 80)
y_pred_ae = (errors > threshold).astype(int)
y_pred_ensemble = ((y_pred_iso == 1) & (y_pred_ae == 1)).astype(int)

print("\nEnsemble Results:")
print(classification_report(y_test, y_pred_ensemble))

# ── 10. SHAP EXPLAINABILITY ───────────────────────────────────
print("\nComputing SHAP values...")
explainer = shap.Explainer(iso_model.predict, X_test[:100])
shap_values = explainer(X_test[:100])

plt.figure()
shap.plots.bar(shap_values, show=False)
plt.tight_layout()
plt.savefig(r'C:\Users\sanga\orbit-shield-ml\outputs\shap_importance.png')
print("SHAP importance plot saved.")

plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig(r'C:\Users\sanga\orbit-shield-ml\outputs\shap_beeswarm.png')
print("SHAP beeswarm plot saved.")

# ── 11. SAVE MODELS ───────────────────────────────────────────
with open(r'C:\Users\sanga\orbit-shield-ml\models\isolation_forest.pkl', 'wb') as f:
    pickle.dump(iso_model, f)
torch.save(model_ae.state_dict(),
           r'C:\Users\sanga\orbit-shield-ml\models\autoencoder.pth')
with open(r'C:\Users\sanga\orbit-shield-ml\models\scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nAll models saved. Project complete.")