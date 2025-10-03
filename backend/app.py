import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from scipy.signal import butter, lfilter

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Initialize Flask App and CORS ---
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- Constants and Pre-defined Mappings ---
SAMPLING_RATE = 128
BANDS = {
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'gamma': (30, 48)
}
CHANNEL_MAP = {
    "Fp1": 0, "Fp2": 16, "F3": 2, "F4": 19, "F7": 4, "F8": 23,
}
ALPHA_BAND_INDEX = 2 # 'alpha' is the 3rd band (index 2)

# --- Load Pre-trained Models and Scaler ---
# Note: These files must be in a 'saved_models' directory.
try:
    VALENCE_MODEL = load_model('saved_models/valence_model_v5_attention.h5')
    AROUSAL_MODEL = load_model('saved_models/arousal_model_v5_attention.h5')
    with open('saved_models/scaler_v5_attention.pkl', 'rb') as f:
        SCALER = pickle.load(f)
    print("Models and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading models or scaler: {e}")
    print("Please ensure 'valence_model_v5_attention.h5', 'arousal_model_v5_attention.h5', and 'scaler_v5_attention.pkl' are in the 'saved_models' directory.")
    VALENCE_MODEL = AROUSAL_MODEL = SCALER = None

# --- Feature Extraction and Processing Functions ---

def get_band_filtered_signal(signal, band, fs=SAMPLING_RATE):
    """
    Filters a signal for a specific frequency band using a Butterworth filter.
    """
    low, high = BANDS[band]
    nyq = 0.5 * fs
    b, a = butter(5, [low / nyq, high / nyq], btype='band')
    return lfilter(b, a, signal)

def extract_features_with_asymmetry(eeg_data):
    """
    Extracts Differential Entropy (DE) features and frontal alpha asymmetry.
    The input eeg_data is expected to have shape (n_trials, n_channels, n_samples).
    """
    n_trials, n_channels, _ = eeg_data.shape
    n_bands = len(BANDS)
    
    # Calculate DE features for all bands
    de_features = np.zeros((n_trials, n_channels, n_bands))
    for trial_idx in range(n_trials):
        for channel_idx in range(n_channels):
            raw_signal = eeg_data[trial_idx, channel_idx, :]
            # Use first 3 seconds (3 * 128 = 384 samples) as baseline
            baseline = raw_signal[:384]
            trial_signal = raw_signal[384:]
            
            # Baseline correction
            corrected_signal = trial_signal - np.mean(baseline)
            
            for band_idx, band in enumerate(BANDS):
                filtered_signal = get_band_filtered_signal(corrected_signal, band)
                variance = np.var(filtered_signal)
                # Differential Entropy formula for a Gaussian signal
                de_features[trial_idx, channel_idx, band_idx] = 0.5 * np.log(2 * np.pi * np.e * variance)

    # Calculate frontal alpha asymmetry
    asymmetry_fp = de_features[:, CHANNEL_MAP["Fp1"], ALPHA_BAND_INDEX] - de_features[:, CHANNEL_MAP["Fp2"], ALPHA_BAND_INDEX]
    asymmetry_f34 = de_features[:, CHANNEL_MAP["F3"], ALPHA_BAND_INDEX] - de_features[:, CHANNEL_MAP["F4"], ALPHA_BAND_INDEX]
    asymmetry_f78 = de_features[:, CHANNEL_MAP["F7"], ALPHA_BAND_INDEX] - de_features[:, CHANNEL_MAP["F8"], ALPHA_BAND_INDEX]

    # Reshape asymmetry features to be concatenated
    asymmetry_fp = np.tile(asymmetry_fp.reshape(-1, 1, 1), (1, n_channels, 1))
    asymmetry_f34 = np.tile(asymmetry_f34.reshape(-1, 1, 1), (1, n_channels, 1))
    asymmetry_f78 = np.tile(asymmetry_f78.reshape(-1, 1, 1), (1, n_channels, 1))
    
    # Concatenate DE features with asymmetry features
    final_features = np.concatenate([de_features, asymmetry_fp, asymmetry_f34, asymmetry_f78], axis=2)
    return final_features

def get_suggestion(valence, arousal):
    """
    Provides a clinical suggestion based on predicted valence and arousal levels.
    """
    if valence == 'High' and arousal == 'High':
        return "Patient is in an optimal, energized, and positive state. Proceed with treatment."
    elif valence == 'High' and arousal == 'Low':
        return "Patient is in a calm, positive state. Good for focused treatment. Proceed."
    elif valence == 'Low' and arousal == 'High':
        return "Patient seems stressed or anxious. Suggest relaxation techniques."
    elif valence == 'Low' and arousal == 'Low':
        return "Patient is in a low mood, low energy state. Consider a motivational conversation."
    return "No suggestion available."

# --- Flask Prediction Endpoint ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the file upload, processes the EEG data, and returns mood predictions.
    """
    if VALENCE_MODEL is None or AROUSAL_MODEL is None or SCALER is None:
        return jsonify({"error": "Backend models not loaded. Check server logs."}), 500

    if 'eeg_file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['eeg_file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and file.filename.endswith('.dat'):
        try:
            # Load and process the .dat file
            # DEAP dataset files are pickled with 'latin1' encoding
            subject_data = pickle.load(file, encoding='latin1')
            eeg_signals = subject_data['data'][:, :32, :] # Use first 32 channels (EEG)
            
            # Extract features
            features = extract_features_with_asymmetry(eeg_signals)
            n_trials, n_channels, n_features = features.shape
            
            # Reshape for scaler, apply scaling, and reshape back
            features_reshaped = features.reshape(-1, n_features)
            scaled_features_reshaped = SCALER.transform(features_reshaped)
            scaled_features = scaled_features_reshaped.reshape(n_trials, n_channels, n_features)
            
            # Make predictions
            valence_preds = VALENCE_MODEL.predict(scaled_features)
            arousal_preds = AROUSAL_MODEL.predict(scaled_features)
            
            # Format results
            results = []
            for i in range(len(valence_preds)):
                valence_label = 'High' if valence_preds[i][0] > 0.5 else 'Low'
                arousal_label = 'High' if arousal_preds[i][0] > 0.5 else 'Low'
                suggestion = get_suggestion(valence_label, arousal_label)
                results.append({
                    "trial": i + 1,
                    "valence": valence_label,
                    "arousal": arousal_label,
                    "suggestion": suggestion
                })
            
            return jsonify(results)

        except Exception as e:
            return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500
    
    return jsonify({"error": "Invalid file type. Please upload a .dat file."}), 400

# --- Run the Flask App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
