from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    with open('models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

def train_knn(X_train, y_train):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    with open('models/knn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model
