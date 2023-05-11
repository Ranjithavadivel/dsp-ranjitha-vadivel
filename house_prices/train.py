import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
import joblib
from house_prices.preprocess import prepare_data


# def build_model(data: pd.DataFrame, model_file_path: str) -> dict[str, str]:
#     # Prepare data
#     X, y = prepare_data(data)

#     # Split into training and test sets with a 70/30 split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     # Train a linear regression model
#     model = LinearRegression()
#     model.fit(X_train, y_train)
    

#     # Evaluate the model
#     rmse = calculate_rmse(model, X_test, y_test)

#     # Save the model
#     joblib.dump(model, model_file_path)

#     # Return performance metrics
#     return {'rmse': rmse}
def build_model(data: pd.DataFrame) -> dict[str, str]:
    # Prepare data
    X= prepare_data(data)
    y=data['SalePrice']

    # Split into training and test sets with a 70/30 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    rmse = calculate_rmse(model, X_test, y_test)

    # Save the model
    model_file_path = '../models/model.joblib'
    joblib.dump(model, model_file_path)

    # Return performance metrics
    return {'rmse': rmse}


def calculate_rmse(model: LinearRegression, X: pd.DataFrame, y: pd.Series) -> float:
    y_pred = model.predict(X)
    mse = ((y_pred - y) ** 2).mean()
    rmse = mse ** 0.5
    return rmse