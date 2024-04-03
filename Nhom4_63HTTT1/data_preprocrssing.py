import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=['CHUANDOAN'])
    y = data['CHUANDOAN']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    return X_train, X_test, y_train, y_test