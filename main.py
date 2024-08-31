import torch
import torch.nn as nn
from train import ModelTrainer
from data import DatasetManager
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class StockPriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockPriceLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create a list to hold LSTM layers and LayerNorm layers
        self.lstm_layers = nn.ModuleList()

        # For each LSTM layer
        for _ in range(self.num_layers):
            # Add an LSTM layer
            self.lstm_layers.append(nn.LSTM(input_size, hidden_size, batch_first=True))
            # Add a LayerNorm layer
            self.lstm_layers.append(nn.LayerNorm(hidden_size))
            # After the first LSTM layer, input size becomes hidden size
            input_size = hidden_size

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h, c = [], []
        for _ in range(self.num_layers):
            h.append(torch.zeros(1, x.size(0), self.hidden_size).to(x.device))
            c.append(torch.zeros(1, x.size(0), self.hidden_size).to(x.device))

        out = x

        # Manually loop through LSTM and LayerNorm layers
        for i in range(self.num_layers):
            # LSTM layer forward pass
            out, (h[i], c[i]) = self.lstm_layers[2 * i](out, (h[i], c[i]))
            # LayerNorm layer forward pass
            out = self.lstm_layers[2 * i + 1](out)

        # Pass the output through the fully connected layers
        out = self.fc1(out[:, -1, :])  # Take the output of the last time step
        out = self.relu(out)
        out = self.fc2(out)

        return out

def plot_stock_chart(df):
    """
    Plot the adjusted close price of a stock from a DataFrame.

    :param df: DataFrame containing 'Date' and 'Adj Close' columns.
    """
    # Ensure 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Set 'Date' as the index
    df.set_index('Date', inplace=True)

    # Plot the adjusted close price
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Adj Close'], label='Adj Close', color='blue')

    # Set the title and labels
    plt.title('Adjusted Close Price of BRK.B')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')

    # Rotate and format the date labels on x-axis
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # # #display the first image in the dataset
    # # display_img(*train_dataset[0])
    # #
    # # #display the first batch
    # # show_batch(train_dl)
    #
    # # print the lengths
    # print(f"Length of Train Data : {len(train_data)}")
    # print(f"Length of Validation Data : {len(val_data)}")
    #
    # # output
    # # Length of Train Data : 12034
    # # Length of Validation Data : 2000

    # use just Adj Close prices for now, but later we may add some other features to allow for better prediction.
    num_features = 1
    # Model parameters
    input_size = num_features  # Number of input features
    hidden_size = 50  # Number of LSTM units to run in parallel
    num_layers = 2  # Number of LSTM layers to stack up
    output_size = 1  # Output size (1 for regression)

    # Initialize the model
    model = StockPriceLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                           output_size=output_size)

    # Loss and optimizer
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # hyper-parameters
    batch_size = 32
    lr = 0.001
    num_epochs = 30
    bc_mod = None
    sequence_length = 50

    scaler = MinMaxScaler(feature_range=(0, 1))

    dataset_path = './resources/BRK_B_stock_price.csv'
    params_dir = './params'

    dataMan = DatasetManager(dataset_path, scaler=scaler)

    df = dataMan.loadDatasetFromFile()

    train_samples = dataMan.makeTensorDataset(df, sequence_length)
    #  since it is an RNN model (i.e.: each output is also an output), training labels are among train_samples
    train_labels = None

    def batch_inputs_callback(batch_inputs):
        return batch_inputs.unsqueeze(-1)

    def batch_labels_callback(batch_labels):
        return batch_labels.unsqueeze(1)

    mt = ModelTrainer(
        model=model, loss_func=loss_func, optimizer=optimizer, train_samples=train_samples, train_labels=train_labels,
        params_dir=params_dir, price_scaler=scaler, batch_size=batch_size, lr=lr, num_epochs=num_epochs, bc_mod=bc_mod,
        batch_inputs_callback=batch_inputs_callback, batch_labels_callback=batch_labels_callback
    )

    mt.train()
