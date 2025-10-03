EEG Mood Prediction Web ApplicationThis project is a full-stack web application that predicts mood (valence and arousal) from EEG data using pre-trained machine learning models. The backend is built with Python and Flask, and the frontend is a single HTML file using vanilla JavaScript and Tailwind CSS.Project Structure/
|-- app.py # The Flask backend server
|-- index.html # The HTML/JS/CSS frontend
|-- saved_models/ # Directory for ML models and scaler
| |-- valence_model_v5_attention.h5
| |-- arousal_model_v5_attention.h5
| |-- scaler_v5_attention.pkl
|-- README.md # This setup guide
PrerequisitesPython 3.8 or newerpip (Python package installer)The three required model/scaler files (.h5, .pkl)Setup Instructions1. Create Project DirectoryFirst, create the main project folder and the required saved_models subdirectory.mkdir eeg-mood-prediction
cd eeg-mood-prediction
mkdir saved_models 2. Place Model FilesPlace your pre-trained model and scaler files into the saved_models directory:valence_model_v5_attention.h5arousal_model_v5_attention.h5scaler_v5_attention.pkl3. Set Up Python EnvironmentIt is highly recommended to use a virtual environment to manage dependencies.# Create a virtual environment
python -m venv venv

# Activate the virtual environment

# On Windows:

.\venv\Scripts\activate

# On macOS/Linux:

source venv/bin/activate 4. Install DependenciesInstall the necessary Python packages using pip.pip install Flask Flask-Cors numpy scipy tensorflow scikit-learn
Note: If you have an M1/M2 Mac, you might need to install tensorflow-macos.5. Run the ApplicationStart the Backend Server:Open your terminal, navigate to the project directory, and run the Flask app.python app.py
The server will start, typically on http://127.0.0.1:5000. You should see a message confirming that the models were loaded successfully.Open the Frontend:In your file explorer, find the index.html file and open it in your web browser (e.g., Chrome, Firefox, Safari).6. How to UseClick the "Choose File" button on the web page.Select a .dat file from the DEAP dataset (e.g., s01.dat).Click the "Predict Mood" button.A loading spinner will appear while the backend processes the file.The results for all 40 trials will be displayed in a table on the page.TroubleshootingCORS Error: If you see a Cross-Origin Resource Sharing (CORS) error in your browser console, ensure the Flask server is running and that Flask-Cors was installed correctly.Model Not Found Error: If the server logs show an error loading models, double-check that the saved_models directory is spelled correctly and contains all three required files.500 Internal Server Error: This usually indicates a problem during data processing. Check the Flask server terminal for a more detailed error traceback. This can happen if the input .dat file is not in the expected format.
