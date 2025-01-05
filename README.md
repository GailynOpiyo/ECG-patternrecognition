ECG Signal Analysis with Hybrid Clustering

This project implements an ECG signal analysis pipeline with a focus on preprocessing, pattern recognition, and clustering using a hybrid approach combining k-Shape and DBSCAN algorithms.
Features

    Signal Input: Supports uploading real ECG signals or generating synthetic ones.
    Preprocessing: Bandpass filtering to remove noise and focus on relevant frequency components.
    Pattern Recognition: Clustering ECG signals using a hybrid model (k-Shape + DBSCAN).
    Report Generation: Exports analysis results in Word and PDF formats.
    User Authentication: Login system for secure access.

File Structure

project/
│
├── app.py                       # Main Streamlit application file
├── hybridkshape.pkl             # Trained hybrid k-Shape model
├── hybriddbscan.pkl             # Trained hybrid DBSCAN model
├── appsynthetic_signals.py      # Module to generate synthetic ECG signals
├── appsignal_preprocessing.py   # Module for signal preprocessing
├── outputs/
│   ├── clusters and metrics/    # Directory for clustering outputs and metrics
│   └── reports/                 # Directory for generated reports
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation (this file)

Installation

    Clone the repository:

git clone https://github.com/yourusername/yourproject.git
cd yourproject

Install required dependencies:

pip install -r requirements.txt

Run the application:

    streamlit run app.py

Usage

    Login: Start the application and log in with your credentials.
    Signal Input: Upload a real ECG signal or generate a synthetic signal.
    Preprocessing: Adjust bandpass filter settings to preprocess the signal.
    Pattern Recognition: Run the hybrid clustering model to analyze the signal.
    Report Generation: Generate and download a Word or PDF report of the analysis.

Hybrid Clustering Approach

This project uses:

    k-Shape Clustering: To capture temporal patterns in ECG signals.
    DBSCAN Clustering: To identify noise and refine clusters.

Example Plot:

Dependencies

    Python 3.8+
    Libraries: Streamlit, NumPy, Pandas, Scikit-learn, tslearn, matplotlib, joblib, scipy, reportlab, python-docx

Contribution

Feel free to contribute to this project by:

    Forking the repository.
    Making a pull request with a detailed description of the changes.

License

This project is licensed under the MIT License.
