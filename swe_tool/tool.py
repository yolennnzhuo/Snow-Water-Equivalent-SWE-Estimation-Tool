# Name: Yulin Zhuo 
# Github username: edsml-yz6622

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import rasterio
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap

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
    alpha = np.std(pred) / np.std(real)
    beta = np.mean(pred) / np.mean(real)
    kge = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
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

def plot_time_series(true_values, predictions, dates=None):
    """
    Plots the true and predicted SWE values over time.
    :param true_values: The ground truth SWE values.
    :param predictions: The predicted SWE values.
    :param dates: The dates corresponding to the values. If None, will use integer indices.
    """
    if dates is not None:
        dates = list(dates)  # Ensure dates is a list
    else:
        dates = list(range(len(true_values)))

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(dates, true_values, label="True Values", color='blue')
    ax.plot(dates, predictions, label="Predicted Values", color='red', linestyle='--')
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel('SWE Value')
    ax.set_title("True vs Predicted values over time")
    ax.grid(True)
    plt.tight_layout()  # Adjust the layout for better view
    plt.show()

def evaluate_model(test_loader, model, dataset):
    """
    Evaluate the trained model by calculating MSE, KGE, MAE and R^2 score, and plot the difference between the true values and the predicted values.
    
    :param test_loader: The dataloader contain the test data. 
    :type test_loader: torch.utils.data.DataLoader
    :param model: The trained model.
    :type model: torch.nn.Module
    :param dataset: An object containing methods for inversing the data scaling.
    :type dataset: class object
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


def plot_snow_class():
    """
    Plot the diagram to show the global snow classification according to snow classification scheme introduced by (Liston and Sturm 2021).
    """
    # Open file
    with rasterio.open('/Users/yz6622/Desktop/IRP/dataset/SnowClass_GL_50km_0.50degree_2021_v01.0.tif') as src:
        snow_class_array = src.read(1) 
        snow_class_transform = src.transform 
        snow_class_crs = src.crs  

    cmap = ListedColormap(['red', 'green', 'orange', 'yellow', 'blue', 'cyan'])  

    # Create map
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) 
    ax.set_global() 
    ax.coastlines() 

    # Show data
    img = ax.imshow(snow_class_array, origin='upper', extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree(), cmap=cmap)
    cbar = plt.colorbar(img, ax=ax, orientation='horizontal', pad=0.03)
    cbar.set_label('Snow Class')

    plt.show()
