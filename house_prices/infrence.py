import pandas as pd
import joblib
import sys
from house_prices.preprocess import prepare_data
sys.path.append('..')

def make_predictions(input_data: pd.DataFrame) -> pd.DataFrame:
    
    
    # Load the model and all the data preparation objects (scaler, encoder, etc)
    model = joblib.load('../models/model.joblib')
    X= prepare_data(input_data)

    # Make predictions
    predictions = model.predict(X)

    return pd.Series(predictions)
