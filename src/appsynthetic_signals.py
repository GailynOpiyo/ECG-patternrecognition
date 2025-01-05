import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def generate_synthetic_ecg(hr=75, noise_level=0.05, length=5000, sampling_rate=1000):
    """
    Generate a synthetic ECG signal using a simple sine wave and known QRS characteristics.
    
    :param hr: Heart rate in beats per minute (default is 75).
    :param noise_level: Noise level to add to the signal (default is 0.05).
    :param length: Length of the ECG signal in samples (default is 5000).
    :param sampling_rate: Sampling rate in Hz (default is 1000 Hz).
    :return: ECG signal with specified heart rate, noise, and length.
    """
    # Duration of one heartbeat in seconds
    heart_rate_period = 60 / hr  # Duration in seconds of one beat

    # Time vector for one beat
    time = np.linspace(0, heart_rate_period, int(sampling_rate * heart_rate_period))

    # Synthetic ECG waveform (simplified version with P, QRS, and T waves as sine waves)
    # QRS complex (sharp spike)
    qrs_complex = np.sin(2 * np.pi * 5 * time) * np.exp(-5 * time)  # QRS complex

    # T-wave (smooth curve after QRS)
    t_wave = np.sin(2 * np.pi * 0.5 * time) * np.exp(-0.5 * time)  # T-wave

    # Combine QRS complex and T-wave
    ecg_signal = qrs_complex + t_wave

    # Repeat to generate full ECG signal
    full_ecg = np.tile(ecg_signal, int(np.ceil(length / len(ecg_signal))))[:length]

    # Add noise to the signal
    noise = np.random.normal(0, noise_level, length)  # Generate Gaussian noise
    ecg_with_noise = full_ecg + noise  # Add the noise

    return ecg_with_noise

# Test the function
signal = generate_synthetic_ecg(hr=75, noise_level=0.05, length=5000, sampling_rate=1000)

# Plot the synthetic ECG signal
plt.plot(signal)
plt.title("Synthetic ECG Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
st.pyplot(plt)
