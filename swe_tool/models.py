# Name: Yulin Zhuo 
# Github username: edsml-yz6622

import torch
from torch import nn
from swe_tool import sweDataset

class LSTM(nn.Module):
    """
    A class used to represent a LSTM Model with flexible architectures.

    Attributes:
        num_epochs: The number of epoches runs for training.
        lstm_layers: List of LSTM layers.
        relu: Activation function.
        fc: Linear transformation to the output.

    Methods:
        forward(x): Build to process the input data throught layers and return the outputs.
    """
    def __init__(self, input_dim=1, hidden_dims=[60, 30], num_epochs=60, output_dim=1):
        """
        Initialise the LSTM model class.

        :param input_dim: The dimension of input data for LSTM layer, default is 1.
        :type input_dim: int, optional
        :param hidden_dims: The dimensionality of hidden layers, default is [60, 30].
        :type hidden_dims: list, optional
        :param num_epochs: The number of epoches runs for training, default is 60.
        :type num_epochs: int, optional
        :param output_dim: The output dimension of output data for LSTM layer, default is 1.
        :type output_dim: int, optional        
        """
        super().__init__()

        self.num_epochs = num_epochs
        self.early_stop = False

        layers = []
        for i in range(len(hidden_dims)):
            # Define the input and hidden dimensions
            input_size = input_dim if i == 0 else hidden_dims[i-1]
            output_size = hidden_dims[i]

            # Create LSTM layers and add it to the list
            layers.append(nn.LSTM(input_size, output_size, batch_first = True))

        # Make sure the layers are stored as part of models
        self.lstm_layers = nn.ModuleList(layers)

        # Define linear layer for final transformation
        self.fc = nn.Linear(hidden_dims[-1], output_dim)
        # Activaton function to avoid negative values
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Define the forward method to pass input.

        :param x: The input tensor.
        :type x: torch.Tensor        

        :return: Output tensor.
        :rtype: torch.Tensor
        """
        # Pass input to all layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)

        # Take the last time step only for prediction
        out = self.fc(x[:, -1, :])

        # Activation function to avoid negative output
        out = self.relu(out)

        return out
    
def train_model(model, ts, n_features, train_loader, val_loader, optimiser, criterion, scheduler, is_early_stop=False, threshold=20):
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
    :param is_early_stop: Whether to enable early stopping or not, default is False.
    :type is_early_stop: bool, optional   
    :param threshold: Determine when to stop the training, e.g stop if no improvement in the next 'threshold' 
                      continuous epochs, default is 20 epochs.
    :type threshold: int, optional   

    :return: A tuple containing the training and validation losses for each epoch.
    :rtype: tuple(list, list)
    """
    # Loss list define
    train_losses = []
    val_losses = []

    # Early stopping define
    best_val_loss = 100.0 # set an initial large value
    count = 0

    for epoch in range(model.num_epochs):
        # Training
        model.train()  # Set the model to training mode
        running_train_loss = 0.0

        for i, (X, Y) in enumerate(train_loader):
            X = X.float()
            Y = Y.float()
            # Reset the gradient to zero
            optimiser.zero_grad()
            # Predict
            outputs = model(X.reshape(-1, ts, n_features)) 
            # Calculate loss
            loss = criterion(outputs, Y)
            # Backward, calculate gradient
            loss.backward()
            # Parameters updated
            optimiser.step() 
            # Loss for all the batches
            running_train_loss += loss.item() * X.size(0) 
        # Average loss for the whole training set
        avg_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0

        with torch.no_grad():
            for i, (X, Y) in enumerate(val_loader):
                X = X.float()
                Y = Y.float()
                # Predict
                outputs = model(X.reshape(-1, ts, n_features))
                # Calculate loss
                loss = criterion(outputs, Y)
                # Loss for all the batches
                running_val_loss += loss.item() * X.size(0)
        # Average loss for the whole validation set
        avg_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1}/{model.num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')

        # Decay learning rate updated
        scheduler.step()

        # Check early stopping
        if is_early_stop:
            # If the loss in this epoch is less than the lowest loss, then update the lowest loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                count = 0
            # If the loss in this epoch is more than the lowest loss, then count the times
            else:
                count += 1
            # If no improvement in the next 20 continuous epochs, then stop training
            if count >= threshold:
                model.early_stop = True
                print('Early stopping!')
                break

    return train_losses, val_losses


def train(df=None, df_test=None, train_file=None, test_file=None, var=['HS'], hidden_dims=[60,30], num_epochs=60, 
          step_size=10, gamma=0.5, ts=30, lr=0.001, is_early_stop=True, threshold=20):   
    """
    Trains an LSTM model using the training data and parameters, then plot the loss function, and evaluate it on the 
    testing set.

    Notes:
    This function performs the following steps:
    1. Data pre-processing: Converts dataframes or files into sweData.
    2. Model building: Initialises an LSTM model with the given parameters.
    3. Training: Trains the model using the training data.
    4. Loss plotting: Plots the training and validation losses over epochs.
    5. Evaluation: Evaluates the model performance on the test data.

    :param df: The input data to train the model, default is None. (Either df or train_file must be provided, but not both.)
    :type df: pandas.Dataframe, optional
    :param df_test: The input data to test the model, default is None. (Either df_test or test_file must be provided, but not both.)
    :type df_test: pandas.Dataframe, optional
    :param train_file: The csv file path to load the data for training, default is None.
    :type train_file: str, optional
    :param test_file: The csv file path to load the data for testing, default is None.
    :type test_file: str, optional
    :param var: The features required for training, default is ['HS'].
    :type var: list, optional
    :param hidden_dims: The dimensionality of hidden layers, default is [60, 30].
    :type hidden_dims: list, optional
    :param num_epochs: The number of epoches runs for training, default is 60.
    :type num_epochs: int, optional
    :param step_size: Step size for the learning rate scheduler. Default is 10.
    :type step_size: int, optional
    :param gamma: Factor of learning rate decay. Default is 0.5.
    :type gamma: float, optional
    :param ts: The time sequence length, the number of time steps to be considered in model, default is 30.
    :type ts: int, optional  
    :param lr: The initial learning rate, default is 0.001.
    :type lr: float, optional  
    :param is_early_stop: Whether to use early stopping during training. Default is True.
    :type is_early_stop: bool, optional
    :param threshold: Threshold for early stopping. Default is 20.
    :type threshold: int, optional

    :return: Trained LSTM model, RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), 
             MBE (Mean Bias Error), KGE (Kling-Gupta Efficiency), and R^2 score.
    :rtype: tuple(torch.nn.Module, float, float, float, float, float)
    """
    # Data preprocess
    dataset = sweDataset.sweDataset(df=df, df_test=df_test, train_file=train_file, test_file=test_file, var=var, ts=ts)
    train_loader, val_loader, test_loader = dataset.get_data_loaders() 

    # Build model
    model = models.LSTM(input_dim=len(var), hidden_dims=hidden_dims, num_epochs=num_epochs)    
    criterion = nn.MSELoss()
    optimiser = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimiser, step_size=step_size, gamma=gamma)

    # Train
    train_losses, val_losses = models.train_model(model, ts, len(var), train_loader, val_loader,
                                                  optimiser, criterion, scheduler,is_early_stop, threshold)

    # Plot the loss function
    plot_loss(train_losses, val_losses)

    # Evaluate
    rmse_test, mae_test, mbe_test, kge_test, r2_test, _, _ = evaluate_model(test_loader, model, dataset)
    
    return model, rmse_test, mae_test, mbe_test, kge_test, r2_test 