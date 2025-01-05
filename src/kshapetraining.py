import numpy as np
import pandas as pd
import neurokit2 as nk
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.model_selection import train_test_split
from tslearn.clustering import KShape
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.signal import resample, butter, filtfilt
import ast
import joblib

# Helper: Bandpass filter
def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=1000, order=5):
    """
    Apply a bandpass filter to a signal.

    Parameters:
    - signal: np.array, the raw ECG signal.
    - lowcut: float, the lower cutoff frequency (default=0.5 Hz).
    - highcut: float, the upper cutoff frequency (default=40 Hz).
    - fs: int, the sampling frequency (default=1000 Hz).
    - order: int, the order of the filter (default=5).

    Returns:
    - filtered_signal: np.array, the bandpass-filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Load data
data = pd.read_csv("new_simulated_ecg_dataset.csv")

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

# Step 5: Optimize k-Shape Clustering
def tune_kshape(signals, k_values):
    best_k = None
    best_score = -1
    best_model = None

    # Remove the extra channel dimension (i.e., reshape to (n_samples, n_timestamps))
    signals = signals.reshape(signals.shape[0], signals.shape[1])

    for k in k_values:
        kshape = KShape(n_clusters=k, verbose=True)
        clusters = kshape.fit_predict(signals)  # fit_predict on reshaped signals (2D)
        silhouette = silhouette_score(signals, clusters)  # Calculate silhouette score using 2D shape
        print(f"k={k}, Silhouette Score = {silhouette}")
        if silhouette > best_score:
            best_k = k
            best_score = silhouette
            best_model = kshape
    return best_model, best_k, best_score

# Now, use train_signals_reshaped in k-Shape
k_values = range(2, 20)  # Test different k values
kshape, best_k, best_silhouette = tune_kshape(train_signals_reshaped, k_values)

# Step 6: Validate and Visualize k-Shape Clusters
train_signals_flat = train_signals_reshaped.reshape(train_signals_reshaped.shape[0], -1)
kshape_clusters = kshape.fit_predict(train_signals_flat)  # Use flat signals for clustering
db_index_kshape = davies_bouldin_score(train_signals_flat, kshape_clusters)

print(f"Best k (k-Shape): {best_k}, Silhouette Score = {best_silhouette}")
print(f"Davies-Bouldin Index (k-Shape): {db_index_kshape}")

# Visualize cluster centroids
plt.figure(figsize=(12, 6))
for cluster_id, centroid in enumerate(kshape.cluster_centers_):
    plt.plot(centroid.ravel(), label=f"Cluster {cluster_id}")
plt.title("Cluster Centroids (k-Shape)")
plt.xlabel("Time")
plt.ylabel("ECG Amplitude")
plt.legend()
plt.show()

# Save Models
joblib.dump(kshape, "kshape_model.pkl")

# success message
print("Model training and saving were successful. k-Shape model has been saved as 'kshape_model.pkl'.")

# Save Metrics
with open("cluster_metrics.txt", "w") as f:
    f.write(f"Best k (k-Shape): {best_k}, Silhouette Score = {best_silhouette}\n")
    f.write(f"Davies-Bouldin Index (k-Shape): {db_index_kshape}\n")

joblib.dump(best_k, "best_k.pkl")