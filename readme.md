EEG Mood Prediction Web Application

This project is a full-stack web application that predicts mood (valence and arousal) from EEG data using pre-trained machine learning models.

Backend: Python (Flask)

Frontend: Single-page index.html using vanilla JavaScript + Tailwind CSS

📂 Project Structure
eeg-mood-prediction/
│-- app.py                     # Flask backend server
│-- index.html                 # HTML/JS/CSS frontend
│-- saved_models/              # Directory for ML models and scaler
│   │-- valence_model_v5_attention.h5
│   │-- arousal_model_v5_attention.h5
│   │-- scaler_v5_attention.pkl
│-- README.md                  # This setup guide

✅ Prerequisites

Python 3.8+

pip (Python package installer)

Pre-trained model and scaler files (.h5, .pkl)

⚙️ Setup Instructions
1. Create Project Directory
mkdir eeg-mood-prediction
cd eeg-mood-prediction
mkdir saved_models

2. Place Model Files

Copy the required pre-trained model and scaler files into the saved_models directory:

valence_model_v5_attention.h5
arousal_model_v5_attention.h5
scaler_v5_attention.pkl

3. Set Up Python Environment

It’s recommended to use a virtual environment:

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

4. Install Dependencies

Install required Python packages:

pip install Flask Flask-Cors numpy scipy tensorflow scikit-learn


⚠️ On M1/M2 Macs, replace TensorFlow with:

pip install tensorflow-macos

5. Run the Application

Start the Backend Server:

python app.py


The server will run at: http://127.0.0.1:5000

You should see a confirmation that the models loaded successfully.

Open the Frontend:

Locate the index.html file in your project folder.

Open it in a browser (Chrome, Firefox, or Safari).

🖥️ How to Use

Click "Choose File" on the web page.

Select a .dat file from the DEAP dataset (e.g., s01.dat).

Click "Predict Mood".

A loading spinner will appear during processing.

The predictions for all 40 trials will be displayed in a results table.

🛠️ Troubleshooting

CORS Error:

Ensure Flask server is running.

Confirm Flask-Cors is installed.

Model Not Found Error:

Check that the saved_models/ folder exists.

Verify all three files are present (.h5, .h5, .pkl).

500 Internal Server Error:

Check Flask terminal logs for traceback.

Often caused by an invalid .dat file format.
