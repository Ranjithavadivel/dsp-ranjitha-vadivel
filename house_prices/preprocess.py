from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import joblib

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Separate the target variable from the features
    X = df
    if 'SalePrice' in df.columns:
        X = df.drop('SalePrice', axis=1)
        y = df['SalePrice']

    # Define the continuous and categorical features
    continuous_features = ['LotArea', 'GarageArea']
    categorical_features = ['Street', 'LotShape', 'GarageQual', 'MSZoning', 'KitchenQual']

    # Scale the continuous features
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X[continuous_features])
    scaled_X = pd.DataFrame(scaled_X, columns=continuous_features)
    cols_to_drop = [col for col in ['GarageQual_Ex', 'GarageQual_nan'] if col in scaled_X.columns]
    scaled_X = scaled_X.drop(cols_to_drop, axis=1)


    # Encode the categorical features using one-hot encoding
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_X = encoder.fit_transform(X[categorical_features])
    encoded_X = pd.DataFrame(encoded_X.toarray(), columns=encoder.get_feature_names_out(categorical_features))
    cols_to_drop_x = [col for col in ['GarageQual_Ex', 'GarageQual_nan', 'KitchenQual_nan', 'MSZoning_nan'] if col in encoded_X .columns]
    encoded_X= encoded_X.drop(cols_to_drop_x, axis=1)

    
    # Concatenate the continuous and categorical features
    X = pd.concat([scaled_X, encoded_X], axis=1)
    
    X=X.dropna()
    
    
    return X
