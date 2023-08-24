# Name: Yulin Zhuo 
# Github username: edsml-yz6622

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, TensorDataset, DataLoader
import tool
import torch

class sweDataset(Dataset):
    """
    A class used to prepare the data for training and testing models.

    Attributes:
        df: The input data to train the model. (Either df or train_file must be provided, but not both.)
        df_test: The input data to test the model. (Either df_test or test_file must be provided, but not both.)
        train_file: The file path to load the data for training.
        test_file: The file path to load the data for testing.
        var: The features required for training.
        ts: The time sequence length, the number of time steps to be considered in model.
        scaler_y: The he scaler used to scale the testing target values.

    Methods:
        get_y_scaler(): Get the scaler for target data.
        get_data_loaders(batch_size=32): Get the data loader for training, testing and validation.
        inverse_scale_target(target): Reverse the predicted value back to the original scale by using the scaler from 'get_y_scaler()'.
    """
    def __init__(self, df=None, df_test=None, train_file=None, test_file=None, var=['HS'], ts=30, scaler_y=None):
        """
        Initialise the LSTM model class.

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
        :param ts: The time sequence length, the number of time steps to be considered in model, default is 30.
        :type ts: int, optional        
        :param scaler_y: The scaler used to scale the testing target values, default is None.
        :type scaler_y: sklearn.preprocessing.Scaler, optional       
        """
        super().__init__()
        
        # Define variables
        self.ts = ts
        self.scaler_y = scaler_y
        self.var = var

        # Read training data
        if train_file is not None and df is None:
            self.df = pd.read_csv(train_file)
        elif df is not None and train_file is None: 
            self.df = df
        # Check if neither df nor train_file is provided
        elif df is None and train_file is None:
            raise ValueError("Either input dataframe or file path must be provided.")
        # Check if both df and train_file are provided
        elif df is not None and train_file is not None:
            raise ValueError("The input dataframe and file path are both provided, can only process one.")

        # Sort the training data by the date
        self.df = self.df.sort_values(by="date")
        self.df.reset_index(drop=True, inplace=True)

        if scaler_y is None:
            # Standardise and min-max scaling
            self.df['temperature'], scaler_temp = tool.standardise_data(np.array(self.df['temperature']).reshape(-1,1), 'train')
            self.df[['HS','precipitation','snowfall','solar_radiation']], scaler_minmax = tool.minmax_data(self.df[['HS','precipitation','snowfall','solar_radiation']].values, 'train')
            self.df['station_SWE'], self.scaler_y = tool.minmax_data(np.array(self.df['station_SWE']).reshape(-1,1), 'train')
        
        # Rebuilding data
        self.input_values, self.target_values = tool.rebuild_data(self.df, var, ts)
        self.X_train, self.X_val, self.y_train, self.y_val = tool.split_dataset(self.input_values, self.target_values)
        
        # Read testing data
        if test_file is not None and df_test is None:
            self.df_test = pd.read_csv(test_file)
        elif df_test is not None and test_file is None: 
            self.df_test = df_test
        # Check if neither df nor train_file is provided
        elif df_test is None and test_file is None:
            raise ValueError("Either input dataframe or file path must be provided.")
        # Check if both df_test and test_file are provided
        elif df_test is not None and test_file is not None:
            raise ValueError("The input dataframe and file path are both provided, can only process one.")

        # Sort the testing data by the date
        self.df_test = self.df_test.sort_values(by="date")
        self.df_test.reset_index(drop=True, inplace=True)

        # Apply the same scaling to test set
        self.df_test['temperature'], _ = tool.standardise_data(np.array(self.df_test['temperature']).reshape(-1,1), 'test', scaler_temp)
        self.df_test[['HS','precipitation','snowfall','solar_radiation']], _ = tool.minmax_data(self.df_test[['HS','precipitation','snowfall','solar_radiation']].values, 'test', scaler_minmax)
        self.df_test['station_SWE'], _ = tool.minmax_data(np.array(self.df_test['station_SWE']).reshape(-1,1), 'test', self.scaler_y)

        # Rebuilding testing data
        self.X_test, self.y_test = tool.rebuild_data(self.df_test, var, ts)

    def get_y_scaler(self):
        """
        Get the scaler for target data.

        :return: The scaler for target data.
        :rtype: sklearn.preprocessing.MinMaxScaler
        """
        return self.scaler_y

    def get_data_loaders(self, batch_size=32):
        """
        Get the data loader for training, testing and validation.

        :param batch_size: The batch size in dataloader, default is 32.
        :type batch_size: int, optional

        :return: (train_loader, val_loader, test_loader): The data loader for training, validating and testing.
        :rtype: tuple of torch.utils.data.DataLoader
        """
        # Convert data to torch tensors
        X_train = torch.tensor(self.X_train.reshape(-1,self.ts,len(self.var))).float()
        X_val = torch.tensor(self.X_val.reshape(-1,self.ts,len(self.var))).float()
        y_train = torch.tensor(self.y_train.reshape(-1,1)).float()
        y_val = torch.tensor(self.y_val.reshape(-1,1)).float()
        X_test = torch.tensor(self.X_test.reshape(-1,self.ts,len(self.var))).float()
        y_test = torch.tensor(self.y_test.reshape(-1,1)).float()

        # Create data loaders
        train_data = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
        val_data = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
        test_data = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

        return train_loader, val_loader, test_loader

    def inverse_scale_target(self, target):
        """
        Inverse the predicted value back to the original scale by using the scaler from 'get_y_scaler()'.

        :param target: The target data for inversing.
        :type target: list or numpy.array

        :return: The inversed target data by the scaler for target data.
        :rtype: list or numpy.array
        """
        return self.scaler_y.inverse_transform(target)
            
    