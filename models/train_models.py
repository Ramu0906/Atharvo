import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import os
import sys

# Ensure the scripts directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from preprocessing import preprocess_data

# Define the file path
data_path = r'C:\Users\Acer\OneDrive\Desktop\Atharvo\data\Job_Placement_Data.csv'

# Load your dataset
if not os.path.exists(data_path):
    raise FileNotFoundError(f"The file at {data_path} does not exist.")
    
data = pd.read_csv(data_path)

# Define features and target
y = data['status']

# Load preprocessor
with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Preprocess the features
X_processed = preprocess_data(data, preprocessor)

columns = data.columns[:-1]
# Convert to DataFrame
X = pd.DataFrame(X_processed, columns=columns)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train models
def create_model_pipeline(model):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

# Create models
lr_model = create_model_pipeline(LogisticRegression())
knn_model = create_model_pipeline(KNeighborsClassifier())
rfc_model = create_model_pipeline(RandomForestClassifier())

# Fit models
lr_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
rfc_model.fit(X_train, y_train)

# Save models to pickle files
with open('models/logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('models/knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)

with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rfc_model, f)

print("Models and preprocessor have been trained and saved successfully.")
