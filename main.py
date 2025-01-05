from tensorflow.keras.models import load_model
from xgboost import XGBRegressor
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

file = './Data/NVDA_cleaned.csv'
xgbr_features = ['Close', 'High', 'Low', 'Open', 'Volume', 'Price_200EMA_diff', 'MA_CO_signal_-1', 'MA_CO_signal_0', 'MA_CO_signal_1']
lstm_features = ['Close', 'High', 'Low', 'Open', 'Volume', 'Price_200EMA_diff', 'MA_CO_signal_-1', 'MA_CO_signal_0', 'MA_CO_signal_1', 'T_reg']


def add_future_price(df, shift_days=1):
    df['Next Close'] = df['Close'].shift(-shift_days)
    df['T_reg'] = (df['Next Close'] - df['Close']) / df['Close']
    df['T_cla'] = df['T_reg'].apply(lambda x: 1 if x > 0 else 0)
    return df

def xgbrPred(df_new):
    # print(np.array(df_new)[-1].shape)
    new_data = df_new[xgbr_features].iloc[-1:].copy()
    predictions = xgbr.predict(np.array(new_data))


    return predictions
    # return 1

def lstmPred(df_new):

    def create_sequences(data, lookback=30):
        X = []
        for i in range(len(data) - lookback):
            X.append(data[i:i + lookback])
        return np.array(X)

    new_data = df_new[lstm_features].copy()
    scaler = joblib.load('lstm_scaler.joblib')
    new_data_scaled = scaler.transform(new_data)

    X_new =  create_sequences(new_data_scaled)
    X_new = np.array([X_new[-1]])

    predictions = lstm_model.predict(X_new)
    return predictions


if __name__ == "__main__":
    lstm_model = load_model('lstm.h5')
    lstm_scaler = joblib.load('lstm_scaler.joblib')

    xgbr = XGBRegressor()
    xgbr.load_model('xgb_regressor.model')

    df_new = pd.read_csv(file)

    df_new = add_future_price(df_new, 1)
    df_new = df_new.dropna()

    lstm_hat = lstmPred(df_new)
    print("lstm: ", lstm_hat)

    xgbr_hat = xgbrPred(df_new)
    print("xgbr: ", xgbr_hat)
