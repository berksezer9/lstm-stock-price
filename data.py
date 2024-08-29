import numpy as np
import yfinance as yf
import pandas as pd
import torch
from torch.utils.data import TensorDataset

class DatasetManager():
    def __init__(self, dataset_path, scaler):
        if not isinstance(dataset_path, str) or not dataset_path.endswith('.csv'):
            raise Exception('self.dataset_path must be a string representing a filepath with .csv extension')

        self.dataset_path = dataset_path
        self.scaler = scaler

    def downloadDataset(self):
        # Define the ticker symbol
        ticker = 'BRK-B'

        # Download historical data
        df = yf.download(ticker, start='2000-01-01', end='2024-08-28')

        # save the data to a CSV file
        df = df[['Date', 'Adj Close']]

        # Convert the Date column to datetime format (optional, but good practice)
        df['Date'] = pd.to_datetime(df['Date'])

        # Sort data by date in case it's not already sorted
        df.sort_values('Date', inplace=True)

        return df

    def saveDatasetToFile(self, df):
        df.to_csv(self.dataset_path, index=False)

        return self

    def loadDatasetFromFile(self):
        try:
            # there may be an exception
            return pd.read_csv(self.dataset_path)
        except Exception as e:
            print('Could not load dataset from the following path: ' + self.dataset_path)

            raise e

    def makeTensorDataset(self, df, sequence_length):
        # Normalize the 'Adj Close' prices
        df['Adj Close'] = self.scaler.fit_transform(df[['Adj Close']])

        # Convert data to sequences of fixed length
        def create_sequences(data, seq_length):
            xs, ys = [], []
            for i in range(len(data) - seq_length):
                x = data[i:i + seq_length]
                y = data[i + seq_length]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)


        # Create sequences
        data = df['Adj Close'].values
        X, y = create_sequences(data, sequence_length)

        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # Create a TensorDataset and DataLoader
        return TensorDataset(X, y)