import numpy as np
import pandas as pd
import neurokit2 as nk
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.model_selection import train_test_split
from tslearn.clustering import KShape
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.signal import resample
import ast
import joblib

# Load data
data = pd.read_csv("new_simulated_ecg_dataset.csv")

# Ensure the dataset contains a "Name" column for signal names or IDs
if "Name" not in data.columns:
    data["Name"] = [f"Signal_{i}" for i in range(len(data))]  # Generate signal names if not present

# Step 1: Preprocess Signals
def preprocess_signal(raw_signal):
    try:
        raw_signal = np.array(ast.literal_eval(raw_signal))  # Convert string to array
        cleaned_signal = nk.ecg_clean(raw_signal, sampling_rate=1000)
        return cleaned_signal
    except Exception as e:
        print(f"Error processing signal: {e}")
        return None

data["Cleaned_Signal"] = data["Signal"].apply(preprocess_signal)

# Step 2: Handle Invalid Data
data["Cleaned_Signal"] = data["Cleaned_Signal"].apply(
    lambda x: x if x is not None and not (np.isnan(x).any() or np.isinf(x).any()) else None
)
data = data.dropna(subset=["Cleaned_Signal"])

# Normalize signals for clustering
train_signals_normalized = TimeSeriesScalerMeanVariance().fit_transform(data["Cleaned_Signal"].tolist())

# Step 3: Resample to Reduce Signal Size
train_signals = np.array([resample(signal.ravel(), 250) for signal in train_signals_normalized])
train_signals = train_signals[:, :, np.newaxis]  # Add channel dimension

# Step 4: Split Data
train_signals, test_signals, train_names, test_names = train_test_split(
    train_signals, data["Name"].values, test_size=0.3, random_state=42
)

# Step 5: Clustering (k-Shape)
kshape = KShape(n_clusters=10, verbose=True)
kshape_clusters = kshape.fit_predict(train_signals)

# Step 6: Define Combined State + Demographic Mapping
states = ["Resting ECG", "Exercise ECG", "Arrhythmia", "Noise/Artifacts", "Tachycardia"]
demographics = ["General", "Younger", "Older", "Child", "Middle-Aged", "Athlete", "Sedentary"]

# Generate state + demographic combinations
cluster_mapping = {}
cluster_index = 0
for state in states:
    for demographic in demographics:
        if cluster_index >= len(np.unique(kshape_clusters)):  # Stop when clusters are exhausted
            break
        cluster_mapping[cluster_index] = f"{state} - {demographic}"
        cluster_index += 1

# Assign cluster labels based on mapping
for cluster_index in range(len(np.unique(kshape_clusters))):
    if cluster_index not in cluster_mapping:
        cluster_mapping[cluster_index] = f"Unmapped Cluster {cluster_index}"

cluster_labels = [cluster_mapping.get(cluster, f"Cluster {cluster}") for cluster in kshape_clusters]

# Step 7: Visualize Cluster Centroids (k-Shape)
plt.figure(figsize=(12, 6))
for cluster_id, centroid in enumerate(kshape.cluster_centers_):
    label = cluster_mapping.get(cluster_id, f"Cluster {cluster_id}")
    plt.plot(centroid.ravel(), label=label)
plt.title("Cluster Centroids (k-Shape)")
plt.xlabel("Time")
plt.ylabel("ECG Amplitude")
plt.legend()
plt.show()

# Step 8: Display Signals in Each Cluster with Names and Labels
for cluster_id, label in cluster_mapping.items():
    plt.figure(figsize=(12, 6))
    cluster_indices = np.where(kshape_clusters == cluster_id)[0]
    cluster_signals = train_signals[cluster_indices][:10]  # Limit to 10 signals for clarity
    cluster_names = train_names[cluster_indices][:10]
    
    for signal, name in zip(cluster_signals, cluster_names):
        plt.plot(signal.ravel(), alpha=0.7, label=name)
    
    plt.title(f"Signals in Cluster: {label}")
    plt.xlabel("Time")
    plt.ylabel("ECG Amplitude")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.show()

# Step 9: Create a Cluster Summary Table
cluster_states = pd.DataFrame({
    "Name": train_names,
    "Cluster": [cluster_mapping.get(cluster, f"Cluster {cluster}") for cluster in kshape_clusters]
})
print("\nCluster Assignments with State + Demographic Labels:")
print(cluster_states)

# Step 10: Real-Time Classification of a New Signal
new_signal = np.array(ast.literal_eval(data["Signal"].iloc[0]))  # Example signal

# Preprocess, normalize, and resample
new_signal_cleaned = nk.ecg_clean(new_signal, sampling_rate=1000)
new_signal_normalized = TimeSeriesScalerMeanVariance().fit_transform([new_signal_cleaned])[0]
new_signal_resampled = resample(new_signal_normalized, 250)

# Predict cluster for new signal
new_signal_kshape_cluster = kshape.predict(new_signal_resampled.reshape(1, -1, 1))[0]
new_signal_label = cluster_mapping.get(new_signal_kshape_cluster, f"Cluster {new_signal_kshape_cluster}")
print(f"New Signal Assigned to Cluster: {new_signal_kshape_cluster}")
print(f"Associated Label: {new_signal_label}")

# Visualize the new signal with cluster assignment
plt.figure(figsize=(10, 4))
plt.plot(new_signal_resampled, label=f"{new_signal_label} (Cluster: {new_signal_kshape_cluster})")
plt.title("New ECG Signal with Cluster Assignment")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Step 11: DBSCAN Noise Detection
dbscan_signals = train_signals.reshape(train_signals.shape[0], -1)  # Flatten signals
dbscan_signals = TimeSeriesScalerMeanVariance().fit_transform(dbscan_signals)  # Normalize

dbscan = DBSCAN(eps=1.0, min_samples=3)
dbscan_clusters = dbscan.fit_predict(dbscan_signals)

noise_indices = np.where(dbscan_clusters == -1)[0]
noise_signals = train_signals[noise_indices]
noise_names = train_names[noise_indices]
print(f"Number of Noise Signals Detected by DBSCAN: {len(noise_signals)}")

# Visualize noise signals with names
plt.figure(figsize=(10, 6))
for signal, name in zip(noise_signals[:10], noise_names[:10]):  # Visualize first 10 noise signals
    plt.plot(signal.ravel(), alpha=0.5, label=name)
plt.title("DBSCAN Noise Signals")
plt.xlabel("Time")
plt.ylabel("ECG Amplitude")
plt.legend(loc='upper right', fontsize='small')
plt.show()

# Save models for reuse
joblib.dump(kshape, "kshape_model.pkl")
joblib.dump(dbscan, "dbscan_model.pkl")
