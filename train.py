import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import time
import os
import matplotlib.pyplot as plt
import numpy as np

class ModelTrainer():
    def __init__(self, model, loss_func, optimizer, train_samples, train_labels, params_dir, price_scaler,
                 batch_size=32, lr=0.001, num_epochs=30, bc_mod=None, batch_inputs_callback=None,
                 batch_labels_callback=None):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_samples = train_samples
        self.train_labels = train_labels
        self.params_dir = params_dir
        self.price_scaler = price_scaler
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_inputs_callback = batch_inputs_callback
        self.batch_labels_callback = batch_labels_callback

        # the batch count modulo. determines how often the model parameters are saved within a training epoch
        # if bc_mod = 8, params will be saved once every 8 batches.
        # if it is set to none, we will save parameters only at the end of an epoch.
        self.bc_mod = bc_mod

        self.gen = torch.Generator().manual_seed(1234)

        # split data into training, validation, test; as 80%, 10%, 10% respectively
        # @ToDo: if there is a separate dataset for test, this data split will not be suitable.
        self.train_size = int(0.8 * len(train_samples))
        self.val_size = int(0.1 * len(train_samples))
        self.test_size = len(train_samples) - self.train_size - self.val_size

        train_data, val_data, test_data = random_split(
            train_samples, [self.train_size, self.val_size, self.test_size], generator=self.gen
        )

        # load the train, validation, and test data into batches. use double the batch size for val and test since they
        # will not need to store gradients.
        self.train_dl = DataLoader(train_data, batch_size)
        self.val_dl = DataLoader(val_data, batch_size * 2)
        self.test_dl = DataLoader(test_data, batch_size * 2)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.loadParams()
    
    def saveModel(self):
        # concatenate name of the params directory with the current timestamp to obtain the file path.
        params_path = f"{self.params_dir}/{str(int(time.time()))}.pt"

        torch.save(self.model.state_dict(), params_path)

        print("Parameters saved successfully.")

    # Define the training function
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        bc = 1

        # for each batch
        for inputs, labels in self.train_dl:
            if self.batch_inputs_callback is not None:
                inputs = self.batch_inputs_callback(inputs)
                
            if self.batch_labels_callback is not None:
                labels = self.batch_labels_callback(labels)
            
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            print("batch count: " + str(bc))

            print("loss: " + str(loss.item()))

            # Accumulate loss
            epoch_loss += loss.item() * inputs.size(0)

            # save model once every 200 samples (num_samples = bc * batch_size)
            if self.bc_mod is not None and bc % self.bc_mod == 1:
                self.saveModel()

            bc += 1

        self.saveModel()

        epoch_loss /= len(self.train_dl.dataset)

        return epoch_loss


    def train(self):
        for epoch in range(self.num_epochs):
            # Train the self.model
            train_loss = self.train_epoch()
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Avg. Train Loss: {train_loss:.4f}')

            # Validate the self.model
            val = self.validate()
            print(f'Avg. Validation Loss: {val["loss"]:.4f}')

            # plot
            self.plotPredictions(val['predictions'], val['actuals'])

        print('Training complete.')

    # use data='test' to use test data, use data='val'= to use validation data.
    def test(self, data='test'):
        self.model.eval()
        test_loss = 0.0
        dl = self.test_dl if data == 'test' else self.val_dl
        predictions = []
        actuals = []

        with torch.no_grad():
            for inputs, labels in dl:
                if self.batch_inputs_callback is not None:
                    inputs = self.batch_inputs_callback(inputs)

                if self.batch_labels_callback is not None:
                    labels = self.batch_labels_callback(labels)

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                batch_loss = self.loss_func(outputs, labels)

                test_loss += batch_loss.item() * inputs.size(0)

                # Save predictions and actual values

                # Get the predictions for the last timestep
                predictions.append(outputs.squeeze().cpu().numpy())
                actuals.append(labels.cpu().numpy())

        test_loss /= len(dl.dataset)

        # Flatten lists to arrays
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)

        # Reverse the scaling (to get original price range)
        predictions = self.price_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals = self.price_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

        return {
            'loss': test_loss,
            'predictions': predictions,
            'actuals': actuals,
        }

    def validate(self):
        return self.test(data='val')
    
    def loadParams(self):
        # we will try loading self.model parameters. if an error occurs we will generate new self.model parameters.
        try:
            files = os.listdir(self.params_dir)

            # if files is empty, raise an exception (which wil be handled by the except clause)
            if len(files) == 0:
                raise Exception('No params file found.')

            # path of the most recent params file
            params_path = f"{self.params_dir}/{max(files)}"

            # Load the parameters from the path
            # (if files is empty, this will raise an exception, which wil be handled by the except clause)
            self.model.load_state_dict(torch.load(params_path, weights_only=True))

            print("Parameters loaded successfully.")
        except Exception:
            print("Failed to load parameters. If you have not saved any parameters yet, this is fine."
                  " Just make sure you have a directory named: " + self.params_dir)

    def plotPredictions(self, predictions, actuals):
        plt.figure(figsize=(14, 7))
        plt.plot(actuals, label='Actual Prices', color='b', linestyle='--', alpha=0.6)
        plt.plot(predictions, label='Predicted Prices', color='orange', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()
