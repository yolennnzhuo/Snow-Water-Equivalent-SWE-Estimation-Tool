import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def inverse_scaling(scaled_data, original_min, original_max):
    return scaled_data * (original_max - original_min) + original_min

def inverse_scale(df):
    """
    Inverse the training and testing data back to the original scale (because the training and testing data are all preprocessed before.)

    :param df: The input data needed to inverse to the original scale.
    :type df: dataframe

    :return: df: The inversed data.
    :rtype: dataframe
    """
    # Inverse scaling
    df['HS'] = inverse_scaling(df['HS'], 0.0, 555.7083333333334)
    df['precipitation'] = inverse_scaling(df['precipitation'], 0, 0.0508635304868221)
    df['snowfall'] = inverse_scaling(df['snowfall'], 0, 0.0508402287960052)
    df['solar_radiation'] = inverse_scaling(df['solar_radiation'], 0, 22261948.0)
    df['temperature'] = inverse_scaling(df['temperature'], -29.88636779785156, 39.6531982421875)
    df['rain'] = inverse_scaling(df['rain'], 0, 0.0488765742629765)
    df['month'] = inverse_scaling(df['month'], 1, 12)
    return df

def minmax_data(data, dtype='train', scaler=None):
    """
    Apply Min-Max Scaler to the training and testing data (to the range 0-1).

    :param data: The input data needed to scale.
    :type data: dataframe
    :param dtype: The type of input data,such as train or test, default is train.
    :type dtype: str, optional
    :param scaler: The scaler used to scale the data, default is None.
    :type scaler: sklearn.preprocessing.Scaler, optional

    :return: (data_norm, scaler): The tuple of scaled data and the corresponding scaler.
    :rtype: tuple
    """
    if dtype == 'train':
        # Create a scaler object
        scaler = MinMaxScaler()
        # Fit and transform the data
        data = np.array(data)
        data_norm = scaler.fit_transform(data)
    elif dtype == 'test':
        data_norm = scaler.transform(data)
    else:
        raise ValueError("'dtype' should either be 'test' or 'train'.")
    
    return data_norm, scaler

def standardise_data(data, dtype='train', scaler=None):
    """
    Apply Strandard Scaler to the training and testing data (to the range -1-1).
    
    :param data: The input data needed to scale.
    :type data: dataframe
    :param dtype: The type of input data,such as train or test, default is train.
    :type dtype: str, optional
    :param scaler: The scaler used to scale the data, default is None.
    :type scaler: sklearn.preprocessing.Scaler, optional

    :return: (data_norm, scaler): The tuple of scaled data and the corresponding scaler.
    :rtype: tuple
    """
    if dtype == 'train':
        # Create a scaler object
        scaler = StandardScaler()
        # Fit and transform the data
        data = np.array(data)
        data_norm = scaler.fit_transform(data)
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
        loc_df = df[df['loc'] == location]  # select the current location only
        loc_df['date'] = pd.to_datetime(loc_df['date'])
        loc_df = loc_df.sort_values('date')  # sort the data
        loc_df.reset_index(drop=True, inplace=True)  # reset the index

        # find the time gap
        time_gaps = loc_df['date'].diff() > pd.Timedelta(1, 'D')
        time_gaps = time_gaps[time_gaps].index.tolist()

        start = 0
        if time_gaps != []:
            for end in time_gaps:
                # Subtract ts from end so that i+ts will not go out of bounds
                for i in range(start, end - ts):
                    input_values.append(loc_df[var][i:i+ts])
                    target_values.append(loc_df['station_SWE'][i+ts])
                start = end + 1

        # last continuous time series
        if len(loc_df) - start >= ts:  # Add this line
            for i in range(start, len(loc_df) - ts):
                input_values.append(loc_df[var][i:i+ts])
                target_values.append(loc_df['station_SWE'][i+ts])

    input_values = np.array(input_values)
    target_values = np.array(target_values)
    
    return input_values, target_values

def split_dataset(X, Y, test_size=0.2,random_state=42):
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
    # reshape the input into one dim
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
    # plot loss function
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
    # plot y_true = y_pred
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_ori, test_pred, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of True vs Predicted Values')
    plt.plot([min(y_test_ori), max(y_test_ori)], [min(y_test_ori), max(y_test_ori)], color='red')  
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
    # evaluation mode
    model.eval()

    test_predictions = []
    y_test_real = []

    # predict on the test data
    with torch.no_grad():
        for i, (X, Y) in enumerate(test_loader):
            X = X.float()
            Y = Y.float()
            outputs = model(X)
            test_predictions.append(outputs.numpy())
            y_test_real.append(Y.numpy())

    # concatenate the list of numpy arrays into one numpy array
    test_predictions = np.concatenate(test_predictions).reshape(-1, 1)
    y_test_real = np.concatenate(y_test_real).reshape(-1, 1)

    # reverse the scaling
    test_pred = dataset.inverse_scale_target(test_predictions)
    y_test_ori = dataset.inverse_scale_target(y_test_real)
    
    # Calculate the rmse kge mae r2
    rmse_test = np.sqrt(mean_squared_error(y_test_ori, test_pred))
    kge_test = kge(y_test_ori.reshape(-1), test_pred.reshape(-1))
    mae_test = mean_absolute_error(y_test_ori, test_pred)
    r2_test = r2_score(y_test_ori, test_pred)

    print('Root Mean Squared Error on Test Data:', rmse_test)
    print('Kling-Gupta efficiency on Test Data:', kge_test)
    print('Mean Absolute Error on Test Data:', mae_test)
    print('R2 Score on Test Data:', r2_test)

    # plot y_true = y_pred
    plot_scatter(y_test_ori, test_pred)
    
def rename_keys(model_path):
    """
    Rename keys in the model's stat dictionary and save the modified version. The function can be used
    when the old structure of model is dfferent from the new structure of model.
    
    :param model_path: The path where the models are stored.
    :type model_path: str
    """
    # Load the original state dict
    state_dict = torch.load(model_path)

    # Create a new dictionary with the renamed keys
    new_state_dict = {}
    for key, value in state_dict.items():
        # Rename the keys here
        new_key = key.replace("lstm1", "lstm_layers.0").replace("lstm2", "lstm_layers.1")
        new_state_dict[new_key] = value

    # Save the modified state dict
    torch.save(new_state_dict, model_path)