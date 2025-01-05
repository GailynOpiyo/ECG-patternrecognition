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

# Step 5: Noise Detection with DBSCAN
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

# Detect noise in train_signals and save the DBSCAN model
labels, noise_indices = detect_noise_dbscan(train_signals_reshaped, eps=0.5, min_samples=5)


# Step 6: User Option to Clean or Cluster As-Is
def clean_or_cluster(train_signals, noise_indices, labels, model=None):
    """
    Optionally clean noisy signals before clustering or cluster as-is.
    """
    # Option 1: Clean noisy signals (remove them from the dataset)
    cleaned_signals = np.delete(train_signals, noise_indices, axis=0)
    print(f"Cleaned signals count: {cleaned_signals.shape[0]}")

    # Option 2: Cluster signals as-is
    if model is None:
        return cleaned_signals  # If the model is not loaded, return cleaned signals for re-clustering
    
    return cleaned_signals, labels

# Option 1: Clean signals before clustering
cleaned_train_signals = clean_or_cluster(train_signals_reshaped, noise_indices, labels)
"""""
# Step 7: Apply the saved k-Shape Model
def apply_pretrained_kshape(signals, model_path="kshape_model.pkl"):
    
   # Apply an already trained k-Shape model to the signals.

    try:
        # Load the pre-trained k-Shape model
        kshape_model = joblib.load(model_path)
        print("Loaded pre-trained k-Shape model.")

        # Reshape the signals for clustering
        signals_flat = signals.reshape(signals.shape[0], -1)  # Flatten the signals

        # Apply the pre-trained k-Shape model to the signals
        clusters = kshape_model.predict(signals_flat)  # Predict clusters
        return clusters, kshape_model
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        return None, None

# Use the pre-trained model for clustering
clusters, kshape_model = apply_pretrained_kshape(cleaned_train_signals)  # Or raw signals if preferred

# Check the clustering results (only if clustering was successful)
if clusters is not None:
    # Evaluate clustering metrics
    db_index_kshape = davies_bouldin_score(cleaned_train_signals.reshape(cleaned_train_signals.shape[0], -1), clusters)
    print(f"Davies-Bouldin Index (k-Shape): {db_index_kshape}")
    
    # Visualize cluster centroids (if needed)
    plt.figure(figsize=(12, 6))
    for cluster_id, centroid in enumerate(kshape_model.cluster_centers_):
        plt.plot(centroid.ravel(), label=f"Cluster {cluster_id}")
    plt.title("Cluster Centroids (k-Shape)")
    plt.xlabel("Time")
    plt.ylabel("ECG Amplitude")
    plt.legend()
    plt.show()
"""