import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

# Define the file path
data_path = r'C:\Users\Acer\OneDrive\Desktop\Atharvo\data\Job_Placement_Data.csv'

# Load your dataset
if not os.path.exists(data_path):
    raise FileNotFoundError(f"The file at {data_path} does not exist.")
    
data = pd.read_csv(data_path)

# Print the column names to find the correct target column
print(data.columns)

# Define features and target
X = data.drop('status', axis=1)
y = data['status']

# Define numerical and categorical columns
numerical_cols = ['ssc_percentage', 'hsc_percentage', 'degree_percentage', 'emp_test_percentage', 'mba_percent']
categorical_cols = ['gender', 'ssc_board', 'hsc_board', 'hsc_subject', 'undergrad_degree', 'work_experience', 'specialisation']

# Define transformers for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), categorical_cols)
    ]
)

preprocessor.fit(X)

with open('models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)


print("Models and preprocessor have been trained and saved successfully.")