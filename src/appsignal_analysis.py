
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def extract_ecg_features(signal, sampling_rate=1000):
    """
    Extract key features like HRV, RR interval, and QT interval using NeuroKit2.
    """
    try:
        # Process ECG signal
        ecg_signals, ecg_info = nk.ecg_process(signal, sampling_rate=sampling_rate)
        # Extract features
        hrv_features = nk.hrv_time(ecg_info['Rpeaks'], sampling_rate=sampling_rate)
        return hrv_features, ecg_signals
    except Exception as e:
        st.error(f"Error in extracting features: {e}")
        return None, None

def detect_peaks(signal, sampling_rate=1000):
    """
    Detect R-peaks using NeuroKit2 and return indices of detected peaks.
    """
    try:
        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
        return rpeaks
    except Exception as e:
        st.error(f"Error in detecting peaks: {e}")
        return None

def compute_statistics(signal):
    """
    Compute basic statistics for the signal.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    variance = np.var(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)

    return {
        "Mean": mean,
        "Standard Deviation": std,
        "Variance": variance,
        "Minimum": min_val,
        "Maximum": max_val,
    }
