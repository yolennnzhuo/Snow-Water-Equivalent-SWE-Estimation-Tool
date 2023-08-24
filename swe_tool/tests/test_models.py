# Name: Yulin Zhuo 
# Github username: edsml-yz6622

import sys
sys.path.insert(0,'.')
import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models  import LSTM, train_model

class TestLSTM():
    @pytest.fixture(autouse=True)
    def setUp(self):
        """
        Setup the input tensor and initialise an LSTM model instance.
        """
        self.model = LSTM(input_dim=2, hidden_dims=[60, 30], num_epochs=5, output_dim=1)
        
        # input dimension: sample size: 100, sequence length: 10, input dimension: 2
        self.X_train = torch.randn(100, 10, 2)
        Y_train = torch.randn(100, 1)
        train_data = TensorDataset(self.X_train, Y_train)
        self.train_loader = DataLoader(train_data, batch_size=16, shuffle=False)

        X_val = torch.randn(30, 10, 2)
        Y_val = torch.randn(30, 1)
        val_data = TensorDataset(X_val, Y_val)
        self.val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=1, gamma=0.7)

    def test_init(self):
        """
        Test the initialisation of the LSTM model.
        """
        assert isinstance(self.model, torch.nn.Module)
        assert len(self.model.lstm_layers) == 2, "Unexpected number of layers"
        assert self.model.num_epochs == 5, "Unexpected number of epochs"

    def test_forward(self):
        """
        Test the forward function of the LSTM model.
        """
        output = self.model(self.X_train)
        assert output.shape == (100, 1), "Unexpected output shape"

    def test_train_model(self):
        """
        Test the train_model function for the LSTM model.
        """
        train_losses, val_losses = train_model(
            self.model, ts=10, n_features=2,
            train_loader=self.train_loader, val_loader=self.val_loader,
            optimiser=self.optimiser, criterion=self.criterion, scheduler=self.scheduler,
            is_early_stop=False
        )
        assert isinstance(train_losses, list), "Unexpected type of train_losses"
        assert isinstance(val_losses, list), "Unexpected type of val_losses"
        assert len(train_losses) == 5, "Unexpected output length of tran_losses"
        assert len(val_losses) == 5, "Unexpected output length of val_losses"

    def test_early_stop(self):
        """
        Test the functionality of early_stopping of training process.
        """
        # Modify the number of epochs for this test
        self.model.num_epochs = 100

        _, _ = train_model(
            self.model, ts=10, n_features=2,
            train_loader=self.train_loader, val_loader=self.val_loader,
            optimiser=self.optimiser, criterion=self.criterion, scheduler=self.scheduler,
            is_early_stop=True
        )
        
        assert self.model.early_stop
    
    def test_decay_lr(self):
        """
        Test the functionality of decay learning rate in training process.
        """
        # Modify the number of epochs for this test
        self.model.num_epochs = 1

        # Initial learning rate
        initial_lr = self.optimiser.param_groups[0]['lr']
        
        # Train for 1 epoch
        _, _ = train_model(
            self.model, ts=10, n_features=2,
            train_loader=self.train_loader, val_loader=self.val_loader,
            optimiser=self.optimiser, criterion=self.criterion, scheduler=self.scheduler,
            is_early_stop=False
        )
        
        # Check if learning rate is decay
        reduced_lr = self.optimiser.param_groups[0]['lr']
        assert np.allclose(reduced_lr, initial_lr * 0.7)


