import torch
import torch.nn as nn
class StockPriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockPriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 25)  # First Dense layer with 25 units
        self.relu = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(25, output_size)  # Output layer with 1 unit for regression

    def forward(self, x):
        # Initialize hidden and cell states for LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Pass the output through the fully connected layers
        out = self.fc1(out[:, -1, :])  # Taking the last time step's output
        out = self.relu(out)
        out = self.fc2(out)

        return out


# use just Adj Close prices for now, but later we may add some other features to allow for better prediction.
num_features = 1
# Model parameters
input_size = num_features  # Number of input features
hidden_size = 50  # Number of LSTM units to run in parallel
num_layers = 2  # Number of LSTM layers to stack up
output_size = 1  # Output size (1 for regression)

# Initialize the model
model = StockPriceLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)