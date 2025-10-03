import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from scipy.signal import butter, lfilter

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- Global Variables & Constants ---
valence_model = None
arousal_model = None
scaler = None

SAMPLING_RATE = 128
BANDS = {
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'gamma': (30, 48)
}
CHANNEL_MAP = {
    "Fp1": 0, "Fp2": 16, "F3": 2, "F4": 19, "F7": 4, "F8": 23,
}
ALPHA_BAND_INDEX = 2

# --- Model Loading ---
@app.before_request
def load_models_once():
    """Load models and scaler only once before the first request."""
    global valence_model, arousal_model, scaler
    if valence_model is None:
        try:
            # Construct paths relative to the current file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, 'saved_models')
            
            # Use compile=False to fix the loading error
            valence_model = load_model(os.path.join(models_dir, 'valence_model_v5_attention.h5'), compile=False)
            arousal_model = load_model(os.path.join(models_dir, 'arousal_model_v5_attention.h5'), compile=False)
            
            with open(os.path.join(models_dir, 'scaler_v5_attention.pkl'), 'rb') as f:
                scaler = pickle.load(f)
            
            print("Models and scaler loaded successfully.")
        except Exception as e:
            print(f"Error loading models or scaler: {e}")
            print("Please ensure 'valence_model_v5_attention.h5', 'arousal_model_v5_attention.h5', and 'scaler_v5_attention.pkl' are in the 'saved_models' directory.")
            # Set to a known state to prevent repeated load attempts
            valence_model = "error" 

# --- Feature Extraction Functions (as provided) ---
def get_band_filtered_signal(signal, band, fs=SAMPLING_RATE):
    low, high = BANDS[band]
    nyq = 0.5 * fs
    b, a = butter(5, [low / nyq, high / nyq], btype='band')
    return lfilter(b, a, signal)

def extract_features_with_asymmetry(eeg_data):
    n_trials, n_channels, _ = eeg_data.shape
    n_bands = len(BANDS)
    de_features = np.zeros((n_trials, n_channels, n_bands))
    for trial_idx in range(n_trials):
        for channel_idx in range(n_channels):
            raw_signal = eeg_data[trial_idx, channel_idx, :]
            baseline = raw_signal[:384]
            trial_signal = raw_signal[384:]
            corrected_signal = trial_signal - np.mean(baseline)
            for band_idx, band in enumerate(BANDS):
                filtered_signal = get_band_filtered_signal(corrected_signal, band)
                variance = np.var(filtered_signal)
                # DE formula: 0.5 * log(2 * pi * e * variance)
                de_features[trial_idx, channel_idx, band_idx] = 0.5 * np.log(2 * np.pi * np.e * variance)
    
    asymmetry_fp = de_features[:, CHANNEL_MAP["Fp1"], ALPHA_BAND_INDEX] - de_features[:, CHANNEL_MAP["Fp2"], ALPHA_BAND_INDEX]
    asymmetry_f34 = de_features[:, CHANNEL_MAP["F3"], ALPHA_BAND_INDEX] - de_features[:, CHANNEL_MAP["F4"], ALPHA_BAND_INDEX]
    asymmetry_f78 = de_features[:, CHANNEL_MAP["F7"], ALPHA_BAND_INDEX] - de_features[:, CHANNEL_MAP["F8"], ALPHA_BAND_INDEX]

    asymmetry_fp = np.tile(asymmetry_fp.reshape(-1, 1, 1), (1, n_channels, 1))
    asymmetry_f34 = np.tile(asymmetry_f34.reshape(-1, 1, 1), (1, n_channels, 1))
    asymmetry_f78 = np.tile(asymmetry_f78.reshape(-1, 1, 1), (1, n_channels, 1))

    final_features = np.concatenate([de_features, asymmetry_fp, asymmetry_f34, asymmetry_f78], axis=2)
    return final_features

def get_suggestion(valence, arousal):
    if valence == 'High' and arousal == 'High': return "Patient is in an optimal, energized, and positive state. Proceed with treatment."
    elif valence == 'High' and arousal == 'Low': return "Patient is in a calm, positive state. Good for focused treatment. Proceed."
    elif valence == 'Low' and arousal == 'High': return "Patient seems stressed or anxious. Suggest relaxation techniques."
    elif valence == 'Low' and arousal == 'Low': return "Patient is in a low mood, low energy state. Consider a motivational conversation."

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if valence_model is None or arousal_model is None or scaler is None:
        return jsonify({'error': 'Models are not loaded. Please check server logs.'}), 500

    if 'eeg_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['eeg_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Load the .dat file
        raw_data = pickle.load(file, encoding='latin1')
        eeg_data = raw_data['data'][:, :32, :] # Use first 32 channels

        # Extract features
        features = extract_features_with_asymmetry(eeg_data)
        
        # Scale features
        num_trials, num_channels, num_features = features.shape
        features_reshaped = features.reshape(num_trials * num_channels, num_features)
        scaled_features_reshaped = scaler.transform(features_reshaped)
        scaled_features = scaled_features_reshaped.reshape(num_trials, num_channels, num_features)

        # Predict Valence and Arousal
        valence_pred_prob = valence_model.predict(scaled_features)
        arousal_pred_prob = arousal_model.predict(scaled_features)
        
        valence_pred = (valence_pred_prob > 0.5).astype(int)
        arousal_pred = (arousal_pred_prob > 0.5).astype(int)

        # Format results
        results = []
        for i in range(len(valence_pred)):
            valence_label = 'High' if valence_pred[i][0] == 1 else 'Low'
            arousal_label = 'High' if arousal_pred[i][0] == 1 else 'Low'
            suggestion = get_suggestion(valence_label, arousal_label)
            results.append({
                'trial': i + 1,
                'valence': valence_label,
                'arousal': arousal_label,
                'suggestion': suggestion
            })
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

# This is required for Render to bind to the correct port
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
