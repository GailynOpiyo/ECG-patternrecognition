# Import necessary libraries
import neurokit2 as nk
import pandas as pd
import numpy as np

# Function to simulate ECG signals for different states
def simulate_ecg(state, heart_rate=70, duration=10, noise=0.05, variability=0):
    """
    Simulates ECG signals for given state.
    
    Parameters:
    - state (str): The label for the state (e.g., "Rest", "Stress").
    - heart_rate (int): Average heart rate (bpm) for the state.
    - duration (int): Duration of the simulated signal (seconds).
    - noise (float): Noise level in the signal.
    - variability (float): Heart rate variability.
    
    Returns:
    - A dictionary with simulated signal and its label.
    """
    ecg_signal = nk.ecg_simulate(
        duration=duration, heart_rate=heart_rate, noise=noise, heart_rate_std=variability
    )
    return {"Signal": ecg_signal, "Label": state}

# Function to simulate signals for demographics
def modify_for_demographics(signal, demographic="General"):
    """
    Modifies ECG signal based on demographic factors.
    
    Parameters:
    - signal (array): Original ECG signal.
    - demographic (str): Demographic label (e.g., "Younger", "Older").
    
    Returns:
    - Modified ECG signal.
    """
    if demographic == "Older":
        return signal + np.random.normal(0, 0.02, len(signal))
    elif demographic == "Younger":
        return signal - np.random.normal(0, 0.01, len(signal))
    elif demographic == "Child":
        return signal + np.random.normal(0, 0.005, len(signal)) * 1.2
    elif demographic == "Middle-Aged":
        return signal + np.random.normal(0, 0.015, len(signal))
    elif demographic == "Athlete":
        return signal - np.random.normal(0, 0.01, len(signal)) * 1.5
    elif demographic == "Sedentary":
        return signal + np.random.normal(0, 0.02, len(signal)) * 0.8
    else:
        return signal

# Function to create a full dataset with 100 samples per combination
def create_ecg_dataset(num_samples=100):
    """
    Simulates ECG signals for various states and demographics, with 100 samples for each combination,
    and saves them to a CSV file.
    """
    dataset = []

    # Define states and their parameters
    states = [
        {"state": "Rest", "heart_rate": 60, "noise": 0.02, "variability": 0.01},
        {"state": "Stress", "heart_rate": 110, "noise": 0.15, "variability": 0.2},
        {"state": "Exercise", "heart_rate": 150, "noise": 0.1, "variability": 0.05},
        {"state": "Arrhythmia", "heart_rate": 70, "noise": 0.2, "variability": 0.4},
        {"state": "Tachycardia", "heart_rate": 180, "noise": 0.1, "variability": 0.01},
    ]

    # Define demographics
    demographics = ["General", "Younger", "Older", "Child", "Middle-Aged", "Athlete", "Sedentary"]

    # Generate 100 samples for each combination of state and demographic
    for state in states:
        for demographic in demographics:
            for _ in range(num_samples):  # Repeat 100 times for each combination
                # Simulate ECG
                ecg_data = simulate_ecg(
                    state=state["state"],
                    heart_rate=state["heart_rate"],
                    duration=10,
                    noise=state["noise"],
                    variability=state["variability"],
                )

                # Modify signal based on demographics
                modified_signal = modify_for_demographics(
                    ecg_data["Signal"], demographic=demographic
                )

                # Add to dataset
                dataset.append(
                    {
                        "Signal": modified_signal.tolist(),  # Convert signal to list for saving
                        "State": state["state"],
                        "Demographic": demographic,
                    }
                )

    # Convert dataset to DataFrame
    df = pd.DataFrame(dataset)

    # Save to CSV
    df.to_csv("C:/Users/user/ai_env/ICSproject2/simulated_ecg_dataset.csv", index=False)
    print(f"Dataset with {num_samples * len(states) * len(demographics)} samples saved to 'simulated_ecg_dataset.csv'.")

# Run the dataset creation function
create_ecg_dataset()