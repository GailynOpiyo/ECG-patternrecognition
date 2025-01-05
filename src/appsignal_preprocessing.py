import numpy as np
from scipy.signal import butter, filtfilt, resample
import pywt

# Example functions used for preprocessing

def bandpass_filter_custom(signal, lowcut=0.5, highcut=40.0, fs=1000, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

def wavelet_denoising(signal, wavelet='db4', level=2):
    coeffs = pywt.wavedec(signal, wavelet, mode='per')
    coeffs[1:] = [pywt.threshold(c, value=np.std(c)/2, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode='per')

def remove_baseline_wander(signal, sampling_rate=1000):
    cutoff = 0.5  # Cutoff frequency in Hz
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)

def dynamic_resampling(signal, target_rate, original_rate):
    return resample(signal, int(len(signal) * target_rate / original_rate))

def pad_signal_to_length(signal, required_length):
    if len(signal) < required_length:
        return np.pad(signal, (0, required_length - len(signal)), mode="constant")
    else:
        return signal[:required_length]

# Main function to preprocess uploaded data
def preprocess_uploaded_signal(signal, target_length=5001, fs=1000):
    """
    This function preprocesses the uploaded signal to fit the model requirements.
    """
    # Step 1: Resample the signal if necessary to match the target sampling rate
    # (if fs of uploaded data differs from the target fs, resample accordingly)
    signal = dynamic_resampling(signal, target_rate=fs, original_rate=fs)

    # Step 2: Apply Bandpass Filter to remove noise
    signal = bandpass_filter_custom(signal, lowcut=0.5, highcut=40.0, fs=fs)

    # Step 3: Apply Wavelet Denoising
    signal = wavelet_denoising(signal, wavelet='db4', level=2)

    # Step 4: Remove Baseline Wander
    signal = remove_baseline_wander(signal, sampling_rate=fs)

    # Step 5: Pad or Truncate to Match Target Length
    signal = pad_signal_to_length(signal, required_length=target_length)

    # Step 6: Normalize the Signal (optional but recommended)
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    return signal

# Example Usage:
uploaded_signal = np.random.randn(4500)  # Example uploaded signal (length 4500)
processed_signal = preprocess_uploaded_signal(uploaded_signal, target_length=5001, fs=1000)

# Reshape the processed signal to fit the model input (1, 5001, 1)
processed_signal_reshaped = processed_signal.reshape(1, processed_signal.shape[0], 1)

# Now the signal is ready for clustering or prediction
print(processed_signal_reshaped.shape)  # Output should be (1, 5001, 1)
