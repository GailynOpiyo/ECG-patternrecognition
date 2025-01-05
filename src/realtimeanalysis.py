import joblib
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import resample, butter, filtfilt
import threading
import queue
import tkinter as tk

# Helper: Bandpass filter
def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=1000, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Load pre-trained k-Shape model
def load_kshape_model(model_path="kshape_model.pkl"):
    try:
        kshape_model = joblib.load(model_path)
        print("Pre-trained k-Shape model loaded successfully.")
        return kshape_model
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        return None

# Real-time signal processing function
def process_real_time_signal(signal, kshape_model, fs=1000):
    # Step 1: Preprocess signal (apply bandpass filter)
    filtered_signal = bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=fs)
    
    # Step 2: Resample signal (if needed)
    resampled_signal = resample(filtered_signal, 250)  # Resample to 250 Hz

    # Step 3: Normalize and reshape (prepare for model input)
    resampled_signal = resampled_signal.reshape(1, -1)  # Reshape to (1, n_timestamps)

    # Step 4: Apply k-Shape model to classify signal
    cluster = kshape_model.predict(resampled_signal)  # Predict cluster
    return cluster

# Real-time signal generator (simulated)
def signal_generator(queue, stop_event):
    while not stop_event.is_set():
        # Simulate ECG signal (replace with real signal)
        signal = np.random.randn(1000)  # Simulated raw ECG signal
        queue.put(signal)  # Push signal to queue
        time.sleep(1)  # Simulate real-time arrival every second

# Start and stop control for the real-time system
def start_real_time_processing():
    # Load pre-trained model (for real-time usage)
    kshape_model = load_kshape_model("kshape_model.pkl")

    if kshape_model is None:
        return  # Exit if model could not be loaded

    # Create a queue for real-time signal processing
    signal_queue = queue.Queue()

    # Start signal generation in a separate thread
    signal_thread = threading.Thread(target=signal_generator, args=(signal_queue, stop_event))
    signal_thread.start()

    # Real-time processing loop
    while True:
        if not signal_queue.empty():
            signal = signal_queue.get()  # Get the latest signal from the queue

            # Process the signal and classify
            cluster = process_real_time_signal(signal, kshape_model)
            print(f"Cluster for this signal: {cluster[0]}")

            # Display the signal
            plt.plot(signal)
            plt.title(f"Cluster: {cluster[0]}")
            plt.show(block=False)  # Non-blocking display
            plt.pause(1)  # Pause for 1 second before next update

        time.sleep(0.1)  # Small delay to allow for signal processing

# Function to start/stop the signal generation
def toggle_signal_generation():
    if not stop_event.is_set():
        stop_event.set()  # Stop signal generation
        print("Signal generation stopped.")
    else:
        stop_event.clear()  # Start signal generation again
        print("Signal generation started.")
        start_real_time_processing()  # Restart processing after signal generation starts

# Initialize stop_event globally
stop_event = threading.Event()

# Initialize GUI for play/stop control
root = tk.Tk()
root.title("Real-Time Signal Processing Control")

# Create a button for play/stop
toggle_button = tk.Button(root, text="Start Signal Generation", command=toggle_signal_generation)
toggle_button.pack(pady=20)

# Start the GUI loop
root.mainloop()
