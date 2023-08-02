import numpy as np
import pandas as pd
from torch.utils.data import Dataset, TensorDataset, DataLoader
import tool
import torch

class sweDataset(Dataset):
    """
    A class used to prepare the data for training and testing models.

    Attributes:
        train_file: The file path to load the data for training.
        test_file: The file path to load the data for testing.
        var: The features required for training.
        ts: The time sequence length, the number of time steps to be considered in model.

    Methods:
        get_y_scaler(): Get the scaler for target data.
        get_data_loaders(batch_size=32): Get the data loader for training, testing and validation.
        inverse_scale_target(target): Reverse the predicted value back to the original scale by using the scaler from 'get_y_scaler()'.
    """
    def __init__(self, df=None, df_test=None, train_file=None, test_file=None, var=['HS'], ts=30):
        """
        Initialise the LSTM model class.

        :param train_file: The csv file path to load the data for training, default is None.
        :type train_file: str, optional
        :param test_file: The csv file path to load the data for testing, default is None.
        :type test_file: str, optional
        :param var: The features required for training, default is ['HS'].
        :type var: list, optional
        :param ts: The time sequence length, the number of time steps to be considered in model, default is 30.
        :type ts: int, optional        
        """
        # Read data
        if train_file is not None and df is None:
            self.df = pd.read_csv(train_file)
        elif df is not None and train_file is None: 
            self.df = df
        elif df is not None and train_file is not None:
            raise ValueError("The input dataframe and file path are both provided, can only process one.")

        # Inverse the data back to original scale
        self.df = tool.inverse_scale(self.df)

        # Sort the data by the date
        self.df = self.df.sort_values(by="date")
        self.df.reset_index(drop=True, inplace=True)

        # Standardise and min-max scaling
        self.df['temperature'], scaler_temp = tool.standardise_data(np.array(self.df['temperature']).reshape(-1,1), 'train')
        self.df[['HS','precipitation','snowfall','solar_radiation','rain','month']], scaler_minmax = tool.minmax_data(self.df[['HS','precipitation','snowfall','solar_radiation','rain','month']], 'train')
        self.df['station_SWE'], scaler_y = tool.minmax_data(np.array(self.df['station_SWE']).reshape(-1,1), 'train')
        
        # Rebuilding data
        self.input_values, self.target_values = tool.rebuild_data(self.df, var, ts)
        self.X_train, self.X_val, self.y_train, self.y_val = tool.split_dataset(self.input_values, self.target_values)
        
        # Read data
        if test_file is not None and df_test is None:
            self.df_test = pd.read_csv(test_file)
        elif df_test is not None and test_file is None: 
            self.df_test = df_test
        elif test_file is not None and df_test is not None:
            raise ValueError("The input dataframe and file path are both provided, can only process one.")

        # Inverse the data back to original scale
        self.df_test = tool.inverse_scale(self.df_test)

        # Sort the data by the date
        self.df = self.df.sort_values(by="date")
        self.df.reset_index(drop=True, inplace=True)

        # Apply the same scaling to test set
        self.df_test['temperature'], _ = tool.standardise_data(np.array(self.df_test['temperature']).reshape(-1,1), 'test', scaler_temp)
        self.df_test[['HS','precipitation','snowfall','solar_radiation','rain','month']], _ = tool.minmax_data(self.df_test[['HS','precipitation','snowfall','solar_radiation','rain','month']], 'test', scaler_minmax)
        self.df_test['station_SWE'], _ = tool.minmax_data(np.array(self.df_test['station_SWE']).reshape(-1,1), 'test', scaler_y)

        # Rebuilding data
        self.X_test, self.y_test = tool.rebuild_data(self.df_test, var, ts)

        # Define variables
        self.ts = ts
        self.scaler_y = scaler_y
        self.var = var

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

        :param batch_size: The batch size in dataloader, default is 30.
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

        :return: The inversed target data by the scaler for target data .
        :rtype: list or numpy.array
        """
        return self.scaler_y.inverse_transform(target)
            
    