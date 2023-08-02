import torch
from torch import nn

class LSTM(nn.Module):
    """
    A class used to represent a LSTM Model with flexible structure.

    Attributes:
        input_dim: The dimension of input data for LSTM layer.
        hidden_dims: The dimensionality of hidden layers.
        num_epoches: The number of epoches runs for training.
        output_dim: The output dimension of output data for LSTM layer.

    Methods:
        forward(x): Build a LSTM model with specified parameters.
    """
    def __init__(self, input_dim=1, hidden_dims=[60, 30], num_epochs=60, output_dim=1):
        super(LSTM, self).__init__()
        """
        Initialise the LSTM model class.

        :param input_dim: The dimension of input data for LSTM layer, default is 1.
        :type input_dim: int, optional
        :param hidden_dims: The dimensionality of hidden layers, default is [60, 30].
        :type hidden_dims: list, optional
        :param num_epoches: The number of epoches runs for training, default is 60.
        :type num_epoches: int, optional
        :param output_dim: The output dimension of output data for LSTM layer, default is 1.
        :type output_dim: int, optional        
        """
        self.num_epochs = num_epochs

        layers = []
        for i in range(len(hidden_dims)):
            # Define the input and hidden dimensions
            input_size = input_dim if i == 0 else hidden_dims[i-1]
            output_size = hidden_dims[i]
            # Create LSTM layers and add it to the list
            layers.append(nn.LSTM(input_size, output_size, batch_first = True))
        # Make sure the layers are stored as part of models
        self.lstm_layers = nn.ModuleList(layers)
        self.fc = nn.Linear(hidden_dims[-1], output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        # Take the last time step only
        out = self.fc(x[:, -1, :])
        # Activation function to avoid negative output
        out = self.relu(out)
        return out
    
def train_model(model, ts, n_features, train_loader, val_loader, optimiser, criterion, scheduler):
    """
    Train and validate the LSTM model.

    :param model: The model to be trained and validated, such as LSTM.
    :type model: torch.nn.Module
    :param ts: The time sequence length, the number of time steps to be considered in model.
    :type ts: int
    :param n_features: The number of features used in training.
    :type n_features: int
    :param train_loader: The training data loader used to train the model.
    :type train_loader:  torch.utils.data.DataLoader
    :param val_loader: The validation data loader used to validate the model.
    :type val_loader: torch.utils.data.DataLoader
    :param optimiser: The optimiser to update the weights of model during the training, such as Adam.
    :type optimiser: torch.optim.Adam
    :param criterion: The loss function used to calcualte the training and validation loss, such as MSE.
    :type criterion: torch.nn.modules
    :param scheduler: The scheduler used to adjust learning rate
    :type scheduler: torch.optim.lr_scheduler.StepLR.

    :return train_losses: The training loss for each epoch.
    :rtype train_losses: list
    :return val_losses: The validation loss for each epoch.
    :rtype val_losses: list
    """
    train_losses = []
    val_losses = []
    for epoch in range(model.num_epochs):
        # Training
        model.train()  # Set the model to training mode
        running_train_loss = 0.0
        for i, (X, Y) in enumerate(train_loader):
            X = X.float()
            Y = Y.float()
            optimiser.zero_grad()
            outputs = model(X.reshape(-1, ts, n_features)) 
            loss = criterion(outputs, Y)
            loss.backward()
            optimiser.step()
            running_train_loss += loss.item() * X.size(0)
        avg_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0
        with torch.no_grad():
            for i, (X, Y) in enumerate(val_loader):
                X = X.float()
                Y = Y.float()
                outputs = model(X.reshape(-1, ts, n_features))
                loss = criterion(outputs, Y)
                running_val_loss += loss.item() * X.size(0)
        avg_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1}/{model.num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')

        # Decay learning rate
        scheduler.step()

    return train_losses, val_losses