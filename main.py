import torch
import torch.nn as nn

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