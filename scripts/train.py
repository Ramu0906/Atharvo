from preprocessing import preprocess_data
from models import train_logistic_regression, train_knn, train_random_forest
from evaluation import evaluate_model
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

def main():
    data_path = r'C:\Users\Acer\OneDrive\Desktop\Atharvo\data\Job_Placement_Data.csv'
    data = pd.read_csv(data_path)

    # Define features and target
    X = data.drop('status', axis=1)
    y = data['status']

    # Load preprocessor
    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    # Preprocess the features
    X_processed = preprocess_data(X, preprocessor)

    columns = data.columns[:-1]
    # Convert to DataFrame
    X = pd.DataFrame(X_processed, columns=columns)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_train.head(5))
    # Train and evaluate Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test)
    
    # Train and evaluate KNN
    knn_model = train_knn(X_train, y_train)
    evaluate_model(knn_model, X_test, y_test)
    
    # Train and evaluate Random Forest
    rfc_model = train_random_forest(X_train, y_train)
    evaluate_model(rfc_model, X_test, y_test)

if __name__ == '__main__':
    main()