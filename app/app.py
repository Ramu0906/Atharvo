from flask import Flask, request, render_template
import pandas as pd
import pickle
import sys
import os

# Add the root directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)

# Load models and preprocessor
def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

lr_model = load_model('models/logistic_regression_model.pkl')
knn_model = load_model('models/knn_model.pkl')
rfc_model = load_model('models/random_forest_model.pkl')
preprocessor = load_model('models/preprocessor.pkl')

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])

def predict():
    x= ''
    if 'file' not in request.files:
        return 'No file uploaded.', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file.', 400
    
    data = pd.read_csv(file)
    print("Data loaded:")
    print(data.head())  # Print the first few rows for debugging

    try:
        # Preprocess data using the loaded preprocessor
        X = preprocessor.transform(data)
        print(f"Processed feature shape: {X.shape}")  # Debugging output to check shape
        x = print(X[1,:])
        
        print()
        # Make predictions
        lr_prediction = lr_model.predict(X)
        knn_prediction = knn_model.predict(X)
        rfc_prediction = rfc_model.predict(X)
        

    except Exception as e:
        return f"Error in making predictions: {e}", 500
    
    return f"Logistic Regression Prediction: {lr_prediction}\nKNN Prediction: {knn_prediction}\nRandom Forest Prediction: {rfc_prediction}"

if __name__ == '__main__':
    app.run(debug=True)
