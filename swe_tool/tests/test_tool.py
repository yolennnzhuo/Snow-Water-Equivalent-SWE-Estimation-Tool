# Name: Yulin Zhuo 
# Github username: edsml-yz6622

import sys
sys.path.insert(0,'.')

import pytest
import numpy as np
import pandas as pd
from torch import nn
import tool
import torch
import sklearn
from predictLSTM import predictLSTM
from sweDataset import sweDataset

class TestTool():
    @pytest.fixture(autouse=True)
    def setUp(self):
        """
        Setup to initialise variables before running each test.
        """
        self.file = pd.DataFrame({
            'date': pd.date_range('2023-08-23', periods=400),
            'HS': np.random.rand(400),
            'station_SWE': np.random.rand(400),
            'loc': ['A' for i in range(400)],
            'temperature': np.random.rand(400),
            'precipitation': np.random.rand(400),
            'snowfall': np.random.rand(400),
            'solar_radiation': np.random.rand(400)
        })

        self.var = ['HS','station_SWE','temperature','precipitation','snowfall','solar_radiation']
        self.df_file = sweDataset(df=self.file, df_test=self.file, var=['HS'])
        # Create model
        self.model = predictLSTM("/Users/yz6622/Desktop/IRP/models/global_model.pth",type='general')
        # For testing Captum related func
        self.attr = np.array([
            [[1, 2], [3, 4]],
            [[2, 3], [4, 5]]
        ])

    def test_mean_bias_error(self):
        """
        Test the mean absolute error computation.
        """
        y_true = [1,2,3]
        y_pred = [2,3,4]
        out = tool.mean_absolute_error(y_true, y_pred)
        assert out == 1
    
    def test_apply_perturbation(self):
        """
        Test the functionality of applying perturbation to a given value.
        """
        hs_1 = 10
        out = tool.apply_perturbation(hs_1)
        assert hs_1 - 1 <= out <= hs_1 + 1

        hs_2 = 30
        out = tool.apply_perturbation(hs_2)
        assert hs_2 - 1.5 <= out <= hs_2 + 1.5

    def test_minmax_data(self):
        '''
        Test the functionality of MinMax Scale.
        formula: X_scale = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        '''
        data = [[1,2,3],[10,11,12]]
        out, scaler = tool.minmax_data(data, 'train')
        assert np.allclose(out, [[0,0,0],[1,1,1]])
        isinstance(scaler, sklearn.preprocessing.MinMaxScaler)

        out, scaler = tool.minmax_data(data, 'test', scaler)
        assert np.allclose(out, [[0,0,0],[1,1,1]])

    def test_standardise_data(self):
        '''
        Test the functionality of Standard Scale.
        formula: z = (x - u) / s
        '''
        data = [[1,2,3],[9,10,11]]
        out, scaler = tool.standardise_data(data, 'train')
        assert np.allclose(out, [[-1,-1,-1],[1,1,1]])
        assert isinstance(scaler, sklearn.preprocessing.StandardScaler)

        out, scaler = tool.standardise_data(data, 'test', scaler)
        assert np.allclose(out, [[-1,-1,-1],[1,1,1]])

    def test_rebuild_data(self):
        """
        Test the functionality of rebuilding data based on given parameters.
        """
        X, Y = tool.rebuild_data(self.file, self.var, 10)
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert X.shape == (390,10,6)
        assert len(X) == len(Y)

    def test_split_dataset(self):
        """
        Test the functionality of splitting datset to training and validation sets.
        """
        X = self.file[['HS','temperature']]
        Y = self.file['station_SWE']
        X_train, X_val, y_train, y_val = tool.split_dataset(X, Y)
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert X_train.shape == (320,2)
        assert X_val.shape == (80,2)

    def test_kge(self):
        """
        Test the Kling-Gupta Efficiency computation.
        """
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        y_pred_offset = np.array([13, 14, 15, 16, 17])
        out = tool.kge(y_true, y_pred)
        out_offset = tool.kge(y_true, y_pred_offset)

        assert np.allclose(out, 1.0)
        assert out_offset < 1.0
    
    def test_plot_loss(self):
        """
        Test the loss plotting functionality.
        """
        train_losses, val_losses = [1,2,3,4], [2,3,4,5]
        try:
            tool.plot_loss(train_losses, val_losses)
        except Exception as e:
            pytest.fail(f"plot_loss() raised an exception: {e}")

    def test_plot_scatter(self):
        """
        Test the error scatter plotting functionality.
        """
        y_test_ori, test_pred = [1,2,3,4], [2,3,4,5]
        try:
            tool.plot_scatter(y_test_ori, test_pred)
        except Exception as e:
            pytest.fail(f"plot_scatter() raised an exception: {e}")
    
    def test_plot_time_series(self):
        """
        Test the time series plotting functionality.
        """
        true_values, predictions = [1,2,3,4], [2,3,4,5]
        try:
            tool.plot_time_series(true_values, predictions)
        except Exception as e:
            pytest.fail(f"plot_time_series() raised an exception: {e}")

    def test_grid_search(self):
        hyper_para = [1,2]
        hyper_type = "ts"
        results = tool.grid_search(hyper_para, df=self.file, df_test=self.file, var=['HS'], ts=10,
                     hidden_dims=[50], num_epochs=5, step_size=1, gamma=0.5, hyper_type = hyper_type)
        assert isinstance(results, dict)
        assert len(results) == 5

        expected_keys = ["RMSE", "MAE", "MBE", "KGE", "R2"]
        for key in expected_keys:
            assert key in results, f"Expected key '{key}' in results"
            assert isinstance(results[key], list), f"Expected type for '{key}' to be a list"
            assert len(results[key]) == len(hyper_para), f"Expected length {len(hyper_para)} for '{key}' but got {len(results[key])}"

    def test_train(self):
        model, rmse_test, mae_test, mbe_test, kge_test, r2_test = tool.train(df=self.file, df_test=self.file, var=['HS'], 
                                                                             hidden_dims=[50], num_epochs=1, step_size=1, 
                                                                             gamma=0.5, ts=1, is_early_stop=True, threshold=20)
        assert isinstance(model, nn.Module)
        assert isinstance(rmse_test, np.floating)
        assert isinstance(mae_test, np.floating)
        assert isinstance(mbe_test, np.floating)
        assert isinstance(kge_test, np.floating)
        assert isinstance(r2_test, np.floating)

    def test_plot_grid_search(self):
        """
        Test the grid search plotting functionality.
        """
        # Setup
        hyper_para = [1,2]
        hyper_type = "ts"
        results = {
            "RMSE": [1,2],
            "MAE": [1,2],
            "MBE": [1,2],
            "KGE": [1,2],
            "R2": [1,2]
        }
        try:
            tool.plot_grid_search(hyper_para, hyper_type, results)
        except Exception as e:
            pytest.fail(f"plot_data_distribution() raised an exception: {e}")

    def test_evaluate_model(self):
        """
        Test the model evaluation process and metrics computation.
        """
        # Creat dataloader
        dataloader, _, _ = self.df_file.get_data_loaders(batch_size=16)
        # Get output
        rmse, mae, mbe, r2, kge_val, y_test_ori, test_pred = tool.evaluate_model(dataloader, self.model.model, self.df_file)

        assert isinstance(rmse, np.floating)
        assert isinstance(mae, np.floating)
        assert isinstance(mbe,np.floating)
        assert isinstance(r2, np.floating)
        assert isinstance(kge_val, np.floating)
        assert isinstance(y_test_ori, np.ndarray)
        assert isinstance(test_pred, np.ndarray)

    def test_cal_integrated_gradients(self):
        """
        Test the functionality of calculating the Integrated Gradients attribution for the given model and input data.
        """
        X, _ = tool.rebuild_data(self.file, ['HS'], 30)
        X = torch.from_numpy(X).float()
        attr = tool.cal_integrated_gradients(self.model, X)

        assert attr.shape == X.shape, f"Expected output shape {X.shape}, but got {attr.shape}"
        
    def test_cal_attr(self):    
        """
        Test the Integrated Gradients attributions for the given attributions computation.
        """
        attr = torch.tensor(self.attr, dtype=torch.float32)
        expected_mean = torch.tensor([2.5, 3.5])
        expected_sum = torch.tensor([10, 14])
        mean_attr, sum_attr = tool.cal_attr(attr)

        assert torch.all(mean_attr.eq(expected_mean)), f"Expected {expected_mean}, but got {mean_attr}"
        assert torch.all(sum_attr.eq(expected_sum)), f"Expected {expected_sum}, but got {sum_attr}"

    def test_plot_attribution(self):
        """
        Test attribution plotting functionality.
        """
        sum_attr = torch.sum(torch.tensor(self.attr),dim=(0,1))
        try:
            tool.plot_attribution(sum_attr,['0','1'])
        except Exception as e:
            pytest.fail(f"plot_attribution() raised an exception: {e}")

    def test_plot_attr_over_time(self):
        """
        Test attribution plotting over time functionality.
        """
        attr = torch.tensor([[[1, 2], [3, 4]], [[2, 3], [4, 5]]], dtype=torch.float32)

        try:
            tool.plot_attr_over_time(attr, 0)
        except Exception as e:
            pytest.fail(f"plot_attr_over_time() raised an exception: {e}")
        
    def test_t_test(self):
        """
        Test t-test computation functionality.
        """
        model_null = [0,0,0] 
        model_modif = [1,1,1] 
    
        try:
            tool.t_test(model_null,model_modif)
        except Exception as e:
            pytest.fail(f"plot_attr_over_time() raised an exception: {e}")


    def test_plot_data_distribution(self):
        """
        Test the data distribution plotting functionality.
        """
        try:
            tool.plot_data_distribution(self.file[['date','HS']], self.file[['date','station_SWE']])
        except Exception as e:
            pytest.fail(f"plot_data_distribution() raised an exception: {e}")

    def test_plot_snow_class(self):
        """
        Test the snow class plotting functionality.
        """
        try:
            tool.plot_snow_class()
        except Exception as e:
            pytest.fail(f"plot_snow_class() raised an exception: {e}")