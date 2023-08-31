# Name: Yulin Zhuo 
# Github username: edsml-yz6622

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, TensorDataset, DataLoader
from swe_tool import tool
import torch

class sweDataset(Dataset):
    """
    A class used to prepare the data for training and testing models.

    - This class enables users to process data either by passing a CSV file or a direct file path. It offers methods to load, interpolate, 
    scale, reconstruct, and partition data. For additional preprocessing, such as applying perturbation, users can preprocess the data 
    externally and then pass it into this class through a file (df).

    Attributes:
        df: The input data to train the model. (Either df or train_file must be provided, but not both.)
        df_test: The input data to test the model. (Either df_test or test_file must be provided, but not both.)
        train_file: The file path to load the data for training.
        test_file: The file path to load the data for testing.
        var: The features required for training.
        ts: The time sequence length, the number of time steps to be considered in model.
        X_train, X_val, X_test: Input data for training, validation, and testing respectively.
        y_train, y_val, y_test: Target data for training, validation, and testing respectively.
        scaler_temp: Scaler used for 'temperature'.
        scaler_minmax: Scaler used for certain features ('HS', 'precipitation', 'snowfall', 'solar_radiation').
        scaler_y: Scaler used for 'station_SWE' feature.

    Methods:
        load_data(df, file_path): Load data either from a dataframe or a file path.
        scale_train_data(): Scale the training data.
        scale_test_data(): Scale the testing data.
        get_y_scaler(): Get the scaler for target data.
        get_data_loaders(batch_size=32): Get the data loader for training, testing and validation.
        inverse_scale_target(target): Reverse the predicted value back to the original scale by using the scaler from 'get_y_scaler()'.
    """
    def __init__(self, df=None, df_test=None, train_file=None, test_file=None, var=['HS'], ts=30):
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
        """
        super().__init__()
        
        # Define variables
        self.ts = ts
        self.var = var

        # Loading training data
        self.df = self.load_data(df, train_file)
        # Interpolate data
        self.df = self.interpolate_data(self.df, ['station_SWE'])
        # Scaling training data
        self.scale_train_data()
        # Rebuilding training data
        input_values, target_values = tool.rebuild_data(self.df, var, ts)
        self.X_train, self.X_val, self.y_train, self.y_val = tool.split_dataset(input_values, target_values)
        
        # Loading testing data
        self.df_test = self.load_data(df_test, test_file)
        # Interpolate data
        self.df_test = self.interpolate_data(self.df_test, ['station_SWE'])
        # Scaling testing data
        self.scale_test_data()
        # Rebuilding testing data
        self.X_test, self.y_test = tool.rebuild_data(self.df_test, var, ts)

    def load_data(self, df, file_path):
        """
        Load data either from a dataframe or a file path.

        :param df: The input dataframe.
        :type df: pandas.Dataframe
        :param file_path: The csv file path to load the data for training.
        :type file_path: str     

        :return: The output dataframe.
        :rtype: pandas.Dataframe
        """
        if file_path is not None and df is None:
            df = pd.read_csv(file_path)
        elif df is not None and file_path is None: 
            df = df
        elif df is None and file_path is None:
            raise ValueError("Either input dataframe or file path must be provided.")
        elif df is not None and file_path is not None:
            raise ValueError("The input dataframe and file path are both provided, can only process one.")
        
        # set the datetime and sort the values
        df = df.sort_values(by=["loc","date"])
        df.reset_index(drop=True, inplace=True)
        return df
    
    def interpolate_data(self, df, var):
        """
        Interpolate target data linearly.

        :param var: The name of variable needs to be interpolated.
        :type var: list, e.g ['station_SWE']

        :return: The interpolated dataframe.
        :rtype: pandas.Dataframe
        """
        df.index = pd.to_datetime(df.index)
        df[var] = df[var].interpolate(method='time')
        df = df.reset_index(drop=True)
        return df
    
    def scale_train_data(self):
        """
        Scale training data.
        """
        self.df['temperature'], self.scaler_temp = tool.standardise_data(np.array(self.df['temperature']).reshape(-1,1), 'train')
        self.df[['HS','precipitation','snowfall','solar_radiation']], self.scaler_minmax = tool.minmax_data(self.df[['HS','precipitation','snowfall','solar_radiation']].values, 'train')
        self.df['station_SWE'], self.scaler_y = tool.minmax_data(np.array(self.df['station_SWE']).reshape(-1,1), 'train')
        
    def scale_test_data(self):
        """
        Scale testing data.
        """
        self.df_test['temperature'], _ = tool.standardise_data(np.array(self.df_test['temperature']).reshape(-1,1), 'test', self.scaler_temp)
        self.df_test[['HS','precipitation','snowfall','solar_radiation']], _ = tool.minmax_data(self.df_test[['HS','precipitation','snowfall','solar_radiation']].values, 'test', self.scaler_minmax)
        self.df_test['station_SWE'], _ = tool.minmax_data(np.array(self.df_test['station_SWE']).reshape(-1,1), 'test', self.scaler_y)
    
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