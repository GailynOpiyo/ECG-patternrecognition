from scipy.stats import mode
import matplotlib.pyplot as plt
import numpy as np
from tslearn.clustering import KShape
import joblib
import pandas as pd
import neurokit2 as nk
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.signal import resample, butter, filtfilt
from sklearn.cluster import DBSCAN
import ast
import joblib

# Helper: Bandpass filter
def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=1000, order=5):
    """
    Apply a bandpass filter to a signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Load data
data = pd.read_csv("C:/Users/user/ai_env/ICSproject2/data/new_simulated_ecg_dataset.csv")

# Ensure the dataset contains a "Name" column for signal names or IDs
if "Name" not in data.columns:
    data["Name"] = [f"Signal_{i}" for i in range(len(data))]  # Generate signal names if not present

# Step 1: Preprocess Signals
def preprocess_signal(raw_signal):
    try:
        raw_signal = np.array(ast.literal_eval(raw_signal))  # Convert string to array
        filtered_signal = bandpass_filter(raw_signal)  # Apply bandpass filter
        cleaned_signal = nk.ecg_clean(filtered_signal, sampling_rate=1000)  # Baseline correction and detrending
        return cleaned_signal
    except Exception as e:
        print(f"Error processing signal: {e}")
        return None

# Apply preprocessing to all signals
data["Cleaned_Signal"] = data["Signal"].apply(preprocess_signal)

# Step 2: Handle Invalid Data
data["Cleaned_Signal"] = data["Cleaned_Signal"].apply(
    lambda x: x if x is not None and not (np.isnan(x).any() or np.isinf(x).any()) else None
)
data = data.dropna(subset=["Cleaned_Signal"])

# Step 3: Resample and Normalize
train_signals = np.array([resample(signal, 250) for signal in data["Cleaned_Signal"].tolist()])
train_signals_normalized = TimeSeriesScalerMeanVariance().fit_transform(train_signals)

# Ensure that the shape is (n_samples, n_timestamps, 1)
train_signals_reshaped = train_signals_normalized[:, :, np.newaxis]  # Add channel dimension (3D shape)

# Step 4: Split Data
train_signals, test_signals, train_names, test_names = train_test_split(
    train_signals_reshaped, data["Name"].values, test_size=0.3, random_state=42
)

# Step 1: Apply DBSCAN for Noise Detection (including noise)
def detect_noise_dbscan(signals, eps=0.5, min_samples=5):
    """
    Use DBSCAN to detect noise in the ECG signals.
    Returns labels and noise detection status.
    """
    # Flatten the signals to 2D for DBSCAN
    signals_flat = signals.reshape(signals.shape[0], signals.shape[1])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(signals_flat)
    
    # Save the trained DBSCAN model
    joblib.dump(dbscan, 'dbscan_model.pkl')  # Save the DBSCAN model
    
    # -1 represents noise in DBSCAN
    noise_indices = np.where(labels == -1)[0]
    print(f"Noise detected in indices: {noise_indices}")
    
    return labels, noise_indices

labels_dbscan, noise_indices = detect_noise_dbscan(train_signals_reshaped, eps=0.5, min_samples=5)

# Step 2: Apply k-Shape Clustering on Each DBSCAN Cluster
hybrid_clusters = []

# Go through each DBSCAN cluster
for dbscan_cluster_id in np.unique(labels_dbscan):
    # For DBSCAN noise, assign it to a separate "Noise" cluster
    if dbscan_cluster_id == -1:
        hybrid_clusters.extend(["Noise"] * np.sum(labels_dbscan == dbscan_cluster_id))
        continue
    
    # Get the indices of signals belonging to the current DBSCAN cluster
    cluster_indices = np.where(labels_dbscan == dbscan_cluster_id)[0]
    
    # Apply k-Shape on signals in this DBSCAN cluster
    signals_in_dbscan_cluster = train_signals_reshaped[cluster_indices]
    signals_in_dbscan_cluster_flat = signals_in_dbscan_cluster.reshape(signals_in_dbscan_cluster.shape[0], -1)
    
    # Fit k-Shape on the signals in the current DBSCAN cluster
    best_k=3
    kshape_model = KShape(n_clusters=best_k, verbose=True)
    labels_kshape = kshape_model.fit_predict(signals_in_dbscan_cluster_flat)
    
    # Append the hybrid cluster labels
    for i, kshape_label in enumerate(labels_kshape):
        hybrid_clusters.append(f"DBSCAN-{dbscan_cluster_id}_kShape-{kshape_label}")

# Add hybrid cluster information to DataFrame for easier analysis
data["Hybrid_Cluster"] = hybrid_clusters

# Step 3: Visualize Hybrid Clusters and Centroids
plt.figure(figsize=(15, 8))

# Plot signals grouped by their hybrid cluster
unique_clusters = np.unique(hybrid_clusters)
for cluster in unique_clusters:
    indices = [i for i, hc in enumerate(hybrid_clusters) if hc == cluster]
    for idx in indices:
        plt.plot(
            train_signals_reshaped[idx].ravel(),
            alpha=0.5,
            label=f"{cluster}" if idx == indices[0] else "",  # Label only once
        )

# Highlight noise
noise_indices = np.where(labels_dbscan == -1)[0]
for idx in noise_indices:
    plt.plot(
        train_signals_reshaped[idx].ravel(),
        color="red",
        alpha=0.5,
        label="Noise" if idx == noise_indices[0] else "",  # Label only once
    )

# Plot k-Shape centroids
for cluster_id, centroid in enumerate(kshape_model.cluster_centers_):
    plt.plot(
        centroid.ravel(),
        linewidth=2,
        linestyle="--",
        label=f"kShape Centroid {cluster_id}",
    )

# Title and legend
plt.title("Hybrid Clusters (DBSCAN + k-Shape) and Centroids")
plt.xlabel("Time")
plt.ylabel("ECG Amplitude")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
plt.tight_layout()
plt.show()

# Save Results
data.to_csv("hybrid_clusters.csv", index=False)
joblib.dump(kshape_model, "hybrid_kshape_model.pkl")
joblib.dump(labels_dbscan, "hybrid_dbscan_labels.pkl")
