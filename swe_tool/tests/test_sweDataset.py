# Name: Yulin Zhuo 
# Github username: edsml-yz6622

import sys
sys.path.insert(0,'.')
import os
import pytest
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sweDataset import sweDataset
import sklearn

class TestSweDataset():
    """
    Test class for the functionality of sweDataset.
    """
    @pytest.fixture(autouse=True)
    def setUp(self):
        """
        Setup testing and training datasets and initialise sweDataset instances.
        """
        self.test_file_t = pd.DataFrame({
            'date': pd.date_range('2023-08-23', periods=900),
            'HS': np.random.rand(900),
            'station_SWE': np.random.rand(900),
            'loc': ['B' for i in range(900)],
            'temperature': np.random.rand(900),
            'precipitation': np.random.rand(900),
            'snowfall': np.random.rand(900),
            'solar_radiation': np.random.rand(900)
        })

        self.train_file_t = pd.DataFrame({
            'date': pd.date_range('2000-06-13', periods=1800),
            'HS': np.random.rand(1800),
            'station_SWE': np.random.rand(1800),
            'loc': ['A' for i in range(1800)],
            'temperature': np.random.rand(1800),
            'precipitation': np.random.rand(1800),
            'snowfall': np.random.rand(1800),
            'solar_radiation': np.random.rand(1800)
        })

        self.test_file_t.to_csv('test_file.csv', index=False)
        self.train_file_t.to_csv('train_file.csv', index=False)
        var = ['HS','station_SWE','temperature','precipitation','snowfall','solar_radiation']
        
        self.df_file = sweDataset(df=self.train_file_t, df_test=self.test_file_t,var=var)
        self.df_path = sweDataset(train_file='train_file.csv',test_file='test_file.csv')

        yield # Will run the code before yield first, separate the setup and removing
        # Remove the temporary files
        os.remove('test_file.csv')
        os.remove('train_file.csv')

    # Check the initialisation
    def test_init(self):
        """
        Test the initialisation of the sweDataset.
        """
        # Check the method to load dataframe
        assert isinstance(self.df_file, sweDataset), "Initialisation with file failed."

        assert self.df_file.df.shape == (1800,8), f"Expected shape (1800,8), but got {self.df_file.df.shape}."
        assert len(self.df_file.X_train) == ((1800-30)*0.8), f"Expected length 1416, but got {self.df_file.X_train}."
        assert len(self.df_file.y_train) == ((1800-30)*0.8), f"Expected length 1416, but got {self.df_file.y_train}."
        assert len(self.df_file.X_test) == 870, f"Expected length 870, but got {self.df_file.X_test}."
        assert len(self.df_file.y_test) == 870, f"Expected length 870, but got {self.df_file.y_test}."

        # Check the method to load data by file path
        assert isinstance(self.df_path, sweDataset), "Initialisation with file path failed."

        assert self.df_path.df.shape == (1800,8), f"Expected shape (1800,8), but got {self.df_path.df.shape}."
        assert len(self.df_path.X_train) == ((1800-30)*0.8), f"Expected length 1416, but got {self.df_path.X_train}."
        assert len(self.df_path.y_train) == ((1800-30)*0.8), f"Expected length 1416, but got {self.df_path.y_train}."
        assert len(self.df_path.X_test) == 870, f"Expected length 870, but got {self.df_path.X_test}."
        assert len(self.df_path.y_test) == 870, f"Expected length 870, but got {self.df_path.y_test}."
    
    # Check if both (or none) of the file and path are provided in initialisation
    def test_both_files_provide(self):
        """
        Ensure ValueError is raised if both dataframe and file path are provided.
        """
        with pytest.raises(ValueError, match="The input dataframe and file path are both provided, can only process one."):
            _ = sweDataset(df=self.train_file_t, train_file='train_file.csv')
    
    def test_no_files_provide(self):
        """
        Ensure ValueError is raised if neither dataframe nor file path are provided.
        """
        with pytest.raises(ValueError, match="Either input dataframe or file path must be provided."):
            _ = sweDataset()
    
    # Check the output type of get_y_scaler()
    def test_get_y_scaler(self):
        """
        Test retrieving the target scaler.
        """
        scaler_y_file = self.df_file.get_y_scaler()
        assert isinstance(scaler_y_file, sklearn.preprocessing.MinMaxScaler), "Unexpected scaler_y type."
        scaler_y_path = self.df_path.get_y_scaler()
        assert isinstance(scaler_y_path, sklearn.preprocessing.MinMaxScaler), "Unexpected scaler_y type."

    # Check the output type and batch size of get_data_loaders()
    def check_batch_size(self, loader_train, loader_val, loader_test):
        """
        Check the batch sizes of the dataloaders.
        """
        batches_train = [data for data in loader_train]
        for data in batches_train[:-1]:
            assert data[0].shape[0] == 16, "Batch size mismatch in training data."
        assert batches_train[-1][0].shape[0] <= 16, "Batch size mismatch in training data."

        batches_val = [data for data in loader_val]
        for data in batches_val[:-1]:
            assert data[0].shape[0] == 16, "Batch size mismatch in validation data."
        assert batches_train[-1][0].shape[0] <= 16, "Batch size mismatch in validation data."

        batches_test = [data for data in loader_test]
        for data in batches_test[:-1]:
            assert data[0].shape[0] == 16, "Batch size mismatch in testing data."
        assert batches_train[-1][0].shape[0] <= 16, "Batch size mismatch in testing data."

    def test_get_loaders_file(self):
        """
        Test functionality of get_data_loaders() of the file instance.
        """
        loader_train, loader_val, loader_test = self.df_file.get_data_loaders(batch_size=16)
        # Check batch size
        self.check_batch_size(loader_train, loader_val, loader_test)
        # Check output type
        assert isinstance(loader_train, DataLoader)
        assert isinstance(loader_val, DataLoader)
        assert isinstance(loader_test, DataLoader)

    def test_get_loaders_by_path(self):
        """
        Test functionality of get_data_loaders() of the file path instance.
        """
        loader_train, loader_val, loader_test = self.df_path.get_data_loaders(batch_size=16)
        # Check batch size
        self.check_batch_size(loader_train, loader_val, loader_test)
        # Check output type
        assert isinstance(loader_train, DataLoader)
        assert isinstance(loader_val, DataLoader)
        assert isinstance(loader_test, DataLoader)

    # Check the output type of inverse_scale_target()
    def test_inverse_scale_target(self):
        """
        Test the inverse scaling functionality for target values.
        """
        target = np.array(self.train_file_t['station_SWE']).reshape(-1,1)
        target_transform = self.df_path.inverse_scale_target(target)
        assert isinstance(target_transform, np.ndarray), "Unexpected target type."