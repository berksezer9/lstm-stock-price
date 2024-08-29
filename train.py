import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import time
import os

class ModelTrainer():
    def __init__(self, model, loss_func, optimizer, train_samples, train_labels, params_dir, batch_size=32, lr=0.001, num_epochs=30):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_samples = train_samples
        self.train_labels = train_labels
        self.params_dir = params_dir
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

        self.val_size = 2000
        self.train_size = len(train_samples) - self.val_size

        train_data, val_data = random_split(train_samples, [self.train_size, self.val_size])

        # load the train and validation into batches.
        self.train_dl = DataLoader(train_data, batch_size, shuffle=True)
        self.val_dl = DataLoader(val_data, batch_size * 2)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.self.device)
    
    def saveModel(self):
        # concatenate name of the params directory with the current timestamp to obtain the file path.
        params_path = f"{self.params_dir}/{str(int(time.time()))}.pt"

        torch.save(self.model.state_dict(), params_path)

        print("Parameters saved successfully.")

    # Define the training function
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        bc = 1

        # for each batch
        for inputs, labels in self.train_dl:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Accumulate loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print("batch count: " + str(bc))

            print("loss: " + str(loss.item()))

            # save self.model once every 100 batches
            if bc % 10 == 1:  # print every once in a while
                self.saveModel()

            bc += 1

        epoch_loss = running_loss / len(self.train_dl.dataset)
        epoch_accuracy = 100 * correct / total

        return epoch_loss, epoch_accuracy


    def train(self):
        for epoch in range(self.num_epochs):
            # Train the self.model
            train_loss, train_accuracy = self.train_epoch()
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

            # Validate the self.model
            val_loss, val_accuracy = self.validate()
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        print('Training complete.')

    # Define the validation function
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_dl:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(self.val_dl.dataset)
        val_accuracy = 100 * correct / total

        return val_loss, val_accuracy
    
    def run(self):
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
            print("Failed to load parameters. Make sure you have the './params' dir")