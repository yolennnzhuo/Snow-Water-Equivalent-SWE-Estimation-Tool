# Name: Yulin Zhuo 
# Github username: edsml-yz6622

import models
import sweDataset
import numpy as np
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import rasterio
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.dates as mdates
from captum.attr import IntegratedGradients
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from scipy import stats

def mean_bias_error(y_true, y_pred):
    """
    Calculate the mean bias error.

    :param y_true: The true value of target variable.
    :type y_true: numpy.ndarray
    :param y_pred: The predicted value of target variable.
    :type y_pred: numpy.ndarray

    :return: Mean bias error of the target variable.
    :rtype: numpy.ndarray
    """
    return np.mean(y_pred - y_true)

def apply_perturbation(hs):
    """
    Apply perturbation to the snow depth to simulate the measurement error and do the data augumentation.

    :param hs: The input data (snow depth) need to apply perturbation.
    :type hs: pandas.core.series.Series

    :return: perturbed_depth: The output seires after applying perturbation according to the 2017 WMO's guidelines.
    :rtype: pandas.core.series.Series
    """
    if hs < 20:
        # When snow depth < 20, errors within ±1
        error = np.random.uniform(-1, 1)
    else:
        # When snow depth >= 20, errors within 5% of the observed values
        max_error = hs * 0.05
        error = np.random.uniform(-max_error, max_error)
    
    perturbed_depth = hs + error
    
    return perturbed_depth

def minmax_data(data, dtype='train', scaler=None):
    """
    Apply Min-Max Scaler to the training and testing data (to the range 0-1).

    :param data: The input data needed to scale.
    :type data: dataframe
    :param dtype: The type of input data, choose train or test, default is train.
    :type dtype: str, optional
    :param scaler: The scaler used to scale the data, default is None.
    :type scaler: sklearn.preprocessing.Scaler, optional

    :return: (data_norm, scaler): The tuple of scaled data and the corresponding scaler.
    :rtype: tuple
    """
    # When handling the training data
    if dtype == 'train':
        # Create a scaler object
        scaler = MinMaxScaler()
        # Fit and transform the data
        data = np.array(data)
        data_norm = scaler.fit_transform(data)

    # When input the testing data      
    elif dtype == 'test':
        # Only transform the data
        data_norm = scaler.transform(data)
    else:
        raise ValueError("'dtype' should either be 'test' or 'train'.")
    
    return data_norm, scaler

def standardise_data(data, dtype='train', scaler=None):
    """
    Apply Strandard Scaler to the training and testing data (to the range from -1 to 1).
    
    :param data: The input data needed to scale.
    :type data: dataframe
    :param dtype: The type of input data, choose train or test, default is train.
    :type dtype: str, optional
    :param scaler: The scaler used to scale the data, default is None.
    :type scaler: sklearn.preprocessing.Scaler, optional

    :return: (data_norm, scaler): The tuple of scaled data and the corresponding scaler.
    :rtype: tuple
    """
    # When handling the training data
    if dtype == 'train':
        # Create a scaler object
        scaler = StandardScaler()
        # Fit and transform the data
        data = np.array(data)
        data_norm = scaler.fit_transform(data)
    
    # When input the testing data
    elif dtype == 'test':
        # Only transform the data
        data_norm = scaler.transform(data)
    else:
        raise ValueError("'dtype' should either be 'test' or 'train'.")
    
    return data_norm, scaler

def rebuild_data(df, var, ts=30):
    """
    Rebuild the input data into the required shape of LSTM.

    The input data should contain the following columns (Col names should be the same.):
    - 'loc': Identifying the location.
    - 'date': The timestamp associated with each sample.
    - 'station_SWE': The target values for estimating.

    :param df: The input data needed to rebuild.
    :type df: dataframe
    :param var: The features selected in rebuilding.
    :type var: list
    :param ts: The time sequence length, the number of time steps to be considered in model, default is 30.
    :type ts: int, optinal

    :return: (input_values, target_values): The tuple of input_values(X) and target_values(Y).
    :rtype: tuple
    """
    input_values = []
    target_values = []

    for location in df['loc'].unique():
        loc_df = df[df['loc'] == location].copy() # Select the current location only
        loc_df['date'] = pd.to_datetime(loc_df['date'])
        loc_df = loc_df.sort_values('date')  # Sort the data
        loc_df.reset_index(drop=True, inplace=True)  # Reset the index

        # Find the time gap
        time_gaps = loc_df['date'].diff() > pd.Timedelta(1, 'D')
        time_gaps = time_gaps[time_gaps].index.tolist()

        start = 0
        if time_gaps != []:
            for end in time_gaps:
                # Subtract ts from end so that i+ts will not go out of bounds
                for i in range(start, end - ts):
                    # Check if there is any missing values
                    if not np.isnan(loc_df['station_SWE'][i+ts]) and all([not np.isnan(loc_df[v][i:i+ts]).any() for v in var]):
                        # Randomly apply the perturbation
                        if not (loc_df[var][i:i+ts].sum(axis=0) == 0).all() or np.random.rand() < 0.5:
                            input_values.append(loc_df[var][i:i+ts])
                            target_values.append(loc_df['station_SWE'][i+ts])
                start = end + 1

        # The last continuous time series
        if len(loc_df) - start >= ts: 
            for i in range(start, len(loc_df) - ts):
                # Check if there is any missing values
                if not np.isnan(loc_df['station_SWE'][i+ts]) and all([not np.isnan(loc_df[v][i:i+ts]).any() for v in var]):
                    # Randomly apply the perturbation
                    if not (loc_df[var][i:i+ts].sum(axis=0) == 0).all() or np.random.rand() < 0.5:
                        input_values.append(loc_df[var][i:i+ts])
                        target_values.append(loc_df['station_SWE'][i+ts])

    input_values = np.array(input_values)
    target_values = np.array(target_values)
    
    return input_values, target_values

def split_dataset(X, Y, test_size=0.2, random_state=42):
    """
    Splitting the dataset into training and validation set.
    
    :param X: The input data (X). 
    :type X: dataframe
    :param Y: The target data (Y) 
    :type Y: list
    :param test_size: The proportion to include in the test part, default is 0.2.
    :type test_size: float, optinal
    :param random_state: The seed used to control the shuffling, default is 42.
    :type random_state: int or None, optinal

    :return: (X_train, X_val, y_train, y_val): The tuple of X_train, X_val, y_train, y_val.
    :rtype: tuple
    """
    # Split into train and test sets
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size,random_state=random_state)
    
    return X_train, X_val, y_train, y_val

def kge(real, pred):
    """
    Calculate the value of Kling-Gupta Efficiency.
    
    :param real: The true test data. 
    :type real: numpy.array or list
    :param pred: The predicted test data. 
    :type pred: numpy.array or list

    :return: kge: The calculated value of Kling-Gupta Efficiency.
    :rtype: float
    """
    # Reshape the input into one dim
    r = np.corrcoef(pred, real)[0,1]
    beta = np.std(pred) / np.std(real)
    gamma = np.mean(pred) / np.mean(real)
    kge = 1 - np.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)
    return kge

def plot_loss(train_losses, val_losses):
    """
    Plot the training and validation loss in the training process.
    
    :param train_losses: The calculated training loss. 
    :type train_losses: numpy.array or list
    :param val_losses: The calculated validation loss. 
    :type val_losses: numpy.array or list
    """
    # Plot loss function
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over time")
    plt.legend()
    plt.show()

def plot_scatter(y_test_ori, test_pred):
    """
    Plot the scattler plot to show the difference between the true data and the predicted data.
    
    :param y_test_ori: The true data. 
    :type y_test_ori: numpy.array or list
    :param test_pred: The predicted data.
    :type test_pred: numpy.array or list
    """
    # Plot y_true = y_pred
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_ori, test_pred, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of True vs Predicted Values')
    plt.plot([min(y_test_ori), max(y_test_ori)], [min(y_test_ori), max(y_test_ori)], color='red')  
    plt.show()

def plot_time_series(true_values, predictions_1, predictions_2=None, dates=None):
    """
    Plots the true and predicted SWE values over time.

    :param true_values: The ground truth SWE values.
    :type true_values: numpy.array or list
    :param predictions_1: The predicted SWE values.
    :type predictions_1: numpy.array or list
    :param predictions_2: The second predicted SWE values.
    :type predictions_2: numpy.array or list, optional
    :param dates: The dates corresponding to the values. If None, will use integer indices.
    :type dates: list, optional
    """
    if dates is not None:
        dates = list(dates)  # Ensure dates is a list
        if isinstance(dates[0], str):
            dates = pd.to_datetime(dates)
    else:
        dates = list(range(len(true_values)))

    # Plot the true and predicted values
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(dates, true_values, label="True Values", color='blue')
    ax.plot(dates, predictions_1, label="Predicted Values by MLSTM", color='green', linestyle='--')
    
    if predictions_2 is not None:  # If the third data is provided
        ax.plot(dates, predictions_2, label="Predicted Values by SLSTM", color='red', linestyle='-.')
    
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel('SWE Value')
    ax.set_title("True vs Predicted values over time")
    ax.grid(True)

    # Display the dates monthly 
    if isinstance(dates[0], (str, pd.Timestamp)):
        locator = mdates.MonthLocator()  
        formatter = mdates.DateFormatter('%d-%m-%Y') 
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

def evaluate_model(test_loader, model, dataset):
    """
    Evaluate the trained model by calculating MSE, KGE, MAE and R^2 score, and plot the difference between the true values and the predicted values.
    
    :param test_loader: The dataloader contain the test data. 
    :type test_loader: torch.utils.data.DataLoader
    :param model: The trained model.
    :type model: torch.nn.Module
    :param dataset: An object containing methods for inversing the data scaling.
    :type dataset: class object

    :return: Tuple of RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), 
             MBE (Mean Bias Error), KGE (Kling-Gupta Efficiency), and R^2 score,
             true test data (y_test_ori) and predicted data (test_pred).
    :rtype: tuple(float, float, float, float, float, np.array, np.array)
    """
    # Evaluation mode
    model.eval()

    test_predictions = []
    y_test_real = []

    # Predict on the test data
    with torch.no_grad():
        for i, (X, Y) in enumerate(test_loader):
            X = X.float()
            Y = Y.float()
            outputs = model(X)
            test_predictions.append(outputs.numpy())
            y_test_real.append(Y.numpy())

    # Concatenate the list of numpy arrays into one numpy array
    test_predictions = np.concatenate(test_predictions).reshape(-1, 1)
    y_test_real = np.concatenate(y_test_real).reshape(-1, 1)

    # Reverse the scaling
    test_pred = dataset.inverse_scale_target(test_predictions)
    y_test_ori = dataset.inverse_scale_target(y_test_real)
    
    # Calculate the rmse kge mae r2 mbe
    rmse_test = np.sqrt(mean_squared_error(y_test_ori, test_pred))
    kge_test = kge(y_test_ori.reshape(-1), test_pred.reshape(-1))
    mae_test = mean_absolute_error(y_test_ori, test_pred)
    r2_test = r2_score(y_test_ori, test_pred)
    mbe_test = mean_bias_error(y_test_ori, test_pred) 

    print('Root Mean Squared Error on Test Data:', rmse_test)
    print('Mean Bias Error on Test Data:', mbe_test)
    print('Mean Absolute Error on Test Data:', mae_test)
    print('Kling-Gupta efficiency on Test Data:', kge_test)
    print('R2 Score on Test Data:', r2_test)

    # Plot y_true = y_pred
    plot_scatter(y_test_ori, test_pred)

    return rmse_test, mae_test, mbe_test, r2_test, kge_test, y_test_ori, test_pred


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

def grid_search(hyper_para, df=None, df_test=None, train_file=None, test_file=None, var=['HS'], ts=30, hidden_dims=[50], 
                num_epochs=60, step_size=10, gamma=0.5, lr=0.001, is_early_stop=True, threshold=20, hyper_type = "ts"):
    """
    Conduct grid search over given hyper-parameters and plots the performance metrics.

    Notes:
    1. This function will loop over the provided hyperparameter values, train a model for each, 
       and then plot performance metrics (RMSE, MAE, MBE, KGE, R2) for the models.
    2. If 'hyper_type' is set to "ts", the grid search will be performed over different time series lengths.
       If set to "architecture", the grid search will be performed over different LSTM architectures.

    :param hyper_para: List of hyperparameters values to search over.
    :type hyper_para: list
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
    :param hidden_dims: The dimensionality of hidden layers, default is [60, 30].
    :type hidden_dims: list, optional
    :param num_epochs: The number of epoches runs for training, default is 60.
    :type num_epochs: int, optional
    :param step_size: Step size for the learning rate scheduler. Default is 10.
    :type step_size: int, optional
    :param gamma: Factor of learning rate decay. Default is 0.5.
    :type gamma: float, optional
    :param is_early_stop: Whether to use early stopping during training. Default is True.
    :type is_early_stop: bool, optional
    :param threshold: Threshold for early stopping. Default is 20.
    :type threshold: int, optional
    :param hyper_type: Type of hyperparameters to search over, default is 'ts'.
    :type hyper_type: str, optional
    """
    # Define variables
    results_rmse = []
    results_mae = []
    results_mbe = []
    results_kge = []
    results_r2 = []

    # If wants to do the grid search on time series
    for i in range(len(hyper_para)):
        # Define the hyper-parameter type
        if hyper_type == "ts":
            ts = hyper_para[i]
        elif hyper_type == "architecture":
            hidden_dims = hyper_para[i]
        else:
            raise ValueError("'hyper_type' should either be 'ts' or 'architecture'.") 
        # Train the model
        _, rmse_test, mae_test, mbe_test, kge_test, r2_test = train(df=df, df_test=df_test, train_file=train_file, 
                                                            test_file=test_file, var=var, hidden_dims=hidden_dims, 
                                                            num_epochs=num_epochs, step_size=step_size,
                                                            gamma=gamma, ts=ts, lr=lr, is_early_stop=is_early_stop,
                                                            threshold=threshold)
        # Append results together
        results_rmse.append(rmse_test)
        results_mae.append(mae_test)  
        results_mbe.append(mbe_test)    
        results_kge.append(kge_test)    
        results_r2.append(r2_test)    
        # Convert result into dictionary
        results = {
        "RMSE": results_rmse,
        "MAE": results_mae,
        "MBE": results_mbe,
        "KGE": results_kge,
        "R2": results_r2
    }

    return results

def plot_grid_search(hyper_para, hyper_type, results):
    """
    Plot the results of grid search.
    """
    # Plot the overall diagrams
    fig, axs = plt.subplots(2, 3, figsize=(18, 12)) 
    hyper_para = hyper_para if hyper_type == "ts" else ["[50]", "[60,30]","[70,50,30]","[70,50,30,10]"]
    xlabel = 'Time length' if hyper_type == "ts" else 'Architecture' 

    axs[0, 0].plot(hyper_para, results['RMSE'], label="RMSE", color='tab:blue', marker='o')
    axs[0, 0].set_title('RMSE')
    axs[0, 0].set_xlabel(xlabel)
    axs[0, 0].set_ylabel('RMSE')

    axs[0, 1].plot(hyper_para, results['MAE'], label="MAE", color='tab:orange', marker='o')
    axs[0, 1].set_title('MAE')
    axs[0, 1].set_xlabel(xlabel)
    axs[0, 1].set_ylabel('MAE')

    axs[0, 2].plot(hyper_para, results['MBE'], label="MBE", color='tab:green', marker='o')
    axs[0, 2].set_title('MBE')
    axs[0, 2].set_xlabel(xlabel)
    axs[0, 2].set_ylabel('MBE')

    axs[1, 0].plot(hyper_para, results['KGE'], label="KGE", color='tab:red', marker='o')
    axs[1, 0].set_title('KGE')
    axs[1, 0].set_xlabel(xlabel)
    axs[1, 0].set_ylabel('KGE')

    axs[1, 1].plot(hyper_para, results['R2'], label="R2", color='tab:purple', marker='o')
    axs[1, 1].set_title('R2')
    axs[1, 1].set_xlabel(xlabel)
    axs[1, 1].set_ylabel('R2')

    axs[1, 2].axis('off')

    fig.suptitle('Performance Metrics for Different Time Sequence Length', fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  
    plt.show()

def cal_integrated_gradients(model, X):
    """
    Calculate the Integrated Gradients attribution for the given model and input data.

    :param model: The trained model whose attributions being calculated.
    :type model: torch.nn.Module
    :param X: The input data of which feature attributions being computed.
    :type X: torch.Tensor

    :return: attr, which is the attributions for each feature in the input data.  
    :rtype: torch.Tensor
    """
    ig = IntegratedGradients(model)
    attr, _ = ig.attribute(X, return_convergence_delta=True)
    return attr 

def cal_attr(attr):
    """
    Calculate the mean and sum of Integrated Gradients attributions for the given attributions.

    :param attr: Attributions tensor calculating from Integrated Gradients methods.
    :type attr: numpy.ndarray or a list

    :return: tuple of (mean_attr, sum_attr) - Mean and sum of the attributions across the specified dimensions.  
    :rtype: tuple of (torch.Tensor, torch.Tensor)
    """
    if not isinstance(attr, torch.Tensor):
        attr = torch.tensor(attr)

    mean_attr = torch.mean(attr, dim=(0, 1))
    sum_attr = torch.sum(attr, dim=(0, 1))

    return mean_attr, sum_attr

def plot_attribution(attribution, feature_names, title="Feature Importances", show_percentage=True, shift=6):
    """
    Plot the attribution of each feature.
    """
    # Sort the abs mean value of each features
    sorted_indices = np.argsort(abs(attribution))
    sorted_attribution = attribution[sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]

    # Define colors
    colors = ['red' if val < 0 else 'blue' for val in sorted_attribution]

    # Plot bar
    plt.figure(figsize=(14, 5))
    plt.xlabel("Mean Attribution")
    plt.ylabel("Features")
    plt.title(title)

    # Adding the attribution values next to the bars
    if show_percentage:
        bars = plt.barh(sorted_feature_names, sorted_attribution, color=colors, alpha=0.7, height=0.8)
        for bar in bars:
            width = bar.get_width()
            label_text = f'{width:.2f}'
            if width < 0:
                label_x = width - shift
            else:
                label_x = width + shift

            plt.text(label_x, bar.get_y() + bar.get_height() / 2, label_text,
                     va='center', ha='center', color='black', fontsize=10)
    
    # Add the legend and labels
    handles = [plt.Rectangle((0,0),1,1, color=color, ec="k") for color in ['red', 'blue']]
    labels= ["Negative Attribution", "Positive Attribution"]
    plt.legend(handles, labels, loc='lower right')
    plt.tight_layout()
    plt.show()

def plot_attr_over_time(attr, feature_idx):
    """
    Plot the attribution of one feature ('HS') over time.
    """
    feature_attr = torch.mean((attr[:, :, feature_idx]), dim=(0)).detach().numpy()
    ts = list(range(len(feature_attr)))

    plt.figure(figsize=(10, 5))
    plt.plot(ts, feature_attr)

    plt.xlabel('Time Step')
    plt.ylabel('Attribution')
    plt.title("Feature Attribution over Time")
    plt.show()

def t_test(perf_null, perf_modif):
    t_stat, p_val = stats.ttest_rel(perf_null, perf_modif)

    print(f'T-statistic: {t_stat}')
    print(f'P-value: {p_val}')

    if p_val < 0.05:
        print('p-value is less than 0.05, hence the modification show significant improvement.')
    else:
        print('p-value is larger than 0.05, hence the modification does not show significant improvement.')    

def plot_data_distribution(HS_df, target_values_df):
    """
    Plot the records distribution across months.
    """
    fig, axarr = plt.subplots(1, 3, figsize=(25, 5))

    # (a) snow depth
    HS_df['HS'].plot(kind='hist', ax=axarr[0], bins=30, edgecolor='black')
    axarr[0].set_title('(d) Filtered Snow Depth')
    axarr[0].set_xlabel('Snow Depth in cm')

    # (b) SWE
    target_values_df['station_SWE'].plot(kind='hist', ax=axarr[1], bins=30, edgecolor='black')
    axarr[1].set_title('(e) Filtered SWE')
    axarr[1].set_xlabel('Snow Water Equivalent in cm')

    # (c) month distribution
    target_values_df = target_values_df.copy()
    target_values_df['date'] = pd.to_datetime(target_values_df['date'])
    target_values_df['month'] = target_values_df['date'].dt.month
    monthly_counts = target_values_df.groupby('month').size()
    
    order = list(range(10, 13)) + list(range(1, 10))
    ordered_counts = monthly_counts.loc[order]
    
    ordered_counts.plot(kind='bar', ax=axarr[2], edgecolor='black')
    axarr[2].set_title('Filtered Number of Records by Month')
    axarr[2].set_xlabel('Month of records')
    axarr[2].set_xticks(range(12))
    axarr[2].set_xticklabels(['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'], rotation=45)
    
    plt.tight_layout()
    plt.show()
    
def plot_snow_class(path='../dataset/SnowClass_GL_50km_0.50degree_2021_v01.0.tif', stations_csv='../dataset/stations_loc.csv'):
    """
    Plot the diagram to show the global snow classification according to snow classification scheme introduced by (Liston and Sturm 2021)
    and stations.
    """
    # Open file
    with rasterio.open(path) as src:
        snow_class_array = src.read(1) 
        snow_class_transform = src.transform 
        snow_class_crs = src.crs

    cmap = ListedColormap(['lightcoral', 'lightgreen', 'orange', 'lightyellow', 'lightskyblue', 'pink','plum','cyan'])  

    # Create map
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) 
    ax.set_global() 
    ax.coastlines() 
    
    # Set to display only the region from North America to Europe
    ax.set_extent([-150, 40, 20, 100], crs=ccrs.PlateCarree())

    # Show data
    img = ax.imshow(snow_class_array, origin='upper', extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree(), cmap=cmap)
    
    # Load stations data and plot them
    stations = pd.read_csv(stations_csv)
    ax.scatter(stations['long'], stations['lat'], s=70, c='black', marker='*', transform=ccrs.PlateCarree(), label='Station')

    # Create legend
    legend_colors = ['lightcoral', 'lightgreen', 'orange', 'lightyellow', 'lightskyblue', 'pink','plum','cyan']
    legend_labels = ['1-Tundra', '2-Boreal Forest', '3-Maritime', '4-Ephemeral (includes no snow)', '5-Prairie', '6-Montane Forest', '7-Ice', '8-Ocean']  
    legend_patches = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
    ax.legend(handles=legend_patches + [Patch(color='black', label='Station')], loc='upper right')

    plt.show()