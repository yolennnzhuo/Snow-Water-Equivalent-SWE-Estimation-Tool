# IRP - Snow Water Equivalent(SWE) Estimation Tool

The 'SWE_tool' package has been designed to enable quick and easy modelling of SWE, across the regions where snow class is Tundra, Boreal Forest, Maritime, Prairie, and Montane Forest.

### Contents

<!-- TOC -->
* [About](#about)
* [Installation Guide](#installation-guide)
* [User Instructions](#user-instructions)
* [Documentation](#documentation)
* [Testing](#testing)
* [Authors](#authors)
* [References](#references)
* [License](#license)
<!-- TOC -->

### About

Snow Water Equivalent (SWE) is essential for managing water resources, producing hydropower, preventing floods, and more, especially in places like California where the annual April snowpack water storage is almost twice as large as surface water reservoir storage [1]. Accurate estimation of SWE is crucial for a few reasons. It can first help with forecasting snowmelt water. Snowmelt water is used for agriculture and human consumption by about one sixth of the world's population (1.2 billion people) [2]. Furthermore, if the accurate peak SWE value can be provided, it can also aid in early-warning for the flooding.

### Installation Guide

Before installing ‘swe_tool’, ensuree your environment meets the dependencies below:

- Python > 3.9
- numpy >= 1.13.0
- scipy
- pandas
- matplotlib
- torch
- scikit-learn
- rasterio
- cartopy
- pytest

If you're using conda, you can set up a new environment with the provided command. First, navigate to the tool directory:

```bash
conda env create -f environment.yml
```

After setting up, activate this environment with:

```bash
conda activate swe_estimation
```

Once inside the environment, you can install ‘swe_tool’ with:

```bash
pip install .
```

### User Instructions

The ’swe_tool‘ package consists of multiple Python files, each with its unique functionality. Below are the descriptions and usage instructions for each:

### Pre-requisite:

Make sure you have navigated to the directory where the 'swe_tool' package is located if you haven't installed it. You can do this by using this command:

```bash
cd /example_path/to/swe_tool_directory
```
Please replace '/path/to/swe_tool_directory' to the actual path to 'swe_tool'.

### sweDataset.py

Description: 'sweDataset' is a Python class designed to prepare data for training and testing models.

Attributes:
- df: The input data to train the model. (Either df or train_file must be provided, but not both.)
- df_test: The input data to test the model. (Either df_test or test_file must be provided, but not both.)
- train_file: The file path to load the data for training.
- test_file: The file path to load the data for testing.
- var: The features required for training.
- ts: The time sequence length, the number of time steps to be considered in model.
- scaler_y: The scaler used to scale the testing target values.

Methods:
- get_y_scaler(): Get the scaler for target data.
- get_data_loaders(batch_size=32): Get the data loader for training, testing and validation.
- inverse_scale_target(target): Reverse the predicted value back to the original scale by using the scaler from 'get_y_scaler()'.

How to use:

1. Import the class:
```bash
import sweDataset
```

2. Instantiate the dataset:
- df: Your training dataframe.
- df_test: Your testing dataframe.
- feature_name: Name of the feature column.
- ts: Time sequence length.
```bash
dataset = sweDataset(df=df, df_test=df_test, var=['feature_name'], ts=ts)
```

3. Get data loaders:
```bash
train_loader, val_loader, test_loader = dataset.get_data_loaders(batch_size=32)
```

4. Inverse predicted values:
```bash
inversed_values = dataset.inverse_scale_target(predicted_values)
```

### models.py

Description: 'models' is a Python class designed to provide a flexible LSTM (Long Short-Term Memory) model structure.

**'LSTM' Class**

The 'LSTM' class is a PyTorch-based implementation of a flexible LSTM architecture.

Attributes:
- num_epochs: The number of epochs runs for training, default value is '60'.
- lstm_layers: List of LSTM layers.
- relu: Activation function (Rectified Linear Unit).
- fc: A fully connected linear layer for transformation.

Methods:
- forward(x): Build to process the input data throught layers and return the outputs.

How to use:

1. Import the class:
```bash
from models import LSTM
```

2. Instantiate the model:
- input_dim: Input dimension.
- hidden_dims: Number of neorons within hidden layers, e.g [60,30].
- num_epochs: Number of epochs.
- output_dim: Output dimension (=1).
```bash
model = LSTM(input_dim=input_dim, hidden_dims=hidden_dims, num_epochs=num_epochs, output_dim=1)
```

3. Generate output:
- input_tensor: Tensor of input data.
```bash
output = model(input_tensor)
```
**'train_model'**

The 'train_model' function is responsible for training and validating the LSTM model.

How to use:
1. Import the function
```bash
from models import train_model
```

2. Train the model:
- model: The LSTM model or any model that inherits from torch.nn.Module.
- ts: Time sequence length, representing the number of time steps to be considered in the model.
- n_features: The number of features used in training.
- train_loader: Data loader for training data.
- val_loader: Data loader for validation data.
- optimiser: The optimiser, e.g Adam, used to update model weights.
- criterion: Loss function, e.g., Mean Squared Error (MSE), for computing the training and validation loss.
- scheduler: Learning rate scheduler, e.g., torch.optim.lr_scheduler.StepLR.
- is_early_stop: Boolean to decide whether early stopping is enabled. Default is False.
```bash
train_losses, val_losses = train_model(
    model=model_instance, 
    ts=ts, 
    n_features=n_features, 
    train_loader=train_data_loader, 
    val_loader=val_data_loader, 
    optimiser=optimiser, 
    criterion=loss_function, 
    scheduler=learning_rate_schedule,
    is_early_stop=False
)
```

### predictLSTM.py

Description: The 'predictLSTM' class is designed to predict SWE values using pre-trained LSTM models. You can predict based on different cases such as general, country-specific, or snow class-specific.

**'predictLSTM' Class**

Attributes:
- country_models: A dictionary where store the LSTM models based on countries.
- snowclass_models: A dictionary where store the LSTM models based on snow classes.
- type: The general category of the model.
- sub_type: The further specific category based on the 'type'.
- model: The pre-trained LSTM model.

Methods:
- call: Make the instance to use the appropriate prediction method.
- add_country_model(x): Ensemble all the LSTM models based on countries by adding to dictionary.
- add_snowclass_model(x): Ensemble all the LSTM models based on different snow classes by adding to dictionary.
- predict_by_country(x): Apply the ensemble LSTM models with a specific location to 
                            predict the SWE.
- predict_by_snowclass(x): Apply the ensemble LSTM models with a specific snow class to predict the SWE.
- predict_by_general(x): Apply the general LSTM models to predict the SWE (The model is trained 
                            on the whole dataset - across countries).

How to use:

1. Import the class:
```bash
from predictLSTM import predictLSTM
```

2. Instantiate the model:
- model_path: Path where store the pre-trained LSTM models.
- type: The general category of the model.
- sub_type: The further specific category based on the 'type'.
```bash
model = predictLSTM(model_path, type, sub_type)
```

3. Predicting SWE:
- input_data: Input data required for the model.
```bash
predicted_swe = model(input_data)
```

**Load model**

The 'load_model' function is responsible for loading a pre-trained model.

How to use:
1. Import the function
```bash
from predictLSTM import load_model
```

2. Load model
- model_path: Path where store the pre-trained LSTM models.
- type: The general category of the model.
- sub_type: The further specific category based on the 'type'.
```bash
model = load_model(model_path, type, sub_type)
```

**Ensemble Models**

The 'model_class' function is responsible for ensembling multiple LSTM models.

How to use:
1. Import the function
```bash
from predictLSTM import model_class
```

2. Ensemble models

For Countries:
- country: Name of the country.
- model_path: Path to the pre-trained LSTM model.
```bash
model.add_country_model(country, model_path)
```

For Snow classes:
- snow_class: Snow class category (integer: 1, 2, 3, 5, or 6).
- model_path: Path where store the pre-trained LSTM model.
```bash
model.add_snowclass_model(snow_class, model_path)
```

### tool.py

Description: The tool.py file contains lots of functions designed for different functionalities including metrics calculation, plotting, and model evaluation in snow water equivalent (SWE) estimation.

**'plot_time_series'**

The 'plot_time_series' function is designed to plot the true values and predicted values of SWE over time. .

How to use:

1. Import the function:
```bash
from tool import plot_time_series
```

2. Plot the diagram:
- true_values: A list of actual SWE values.
- predictions: A list of predicted SWE values.
- dates(optional): Dates corresponding to each value. If not provided, the function will default to using integer indices.
```bash
plot_time_series(true_values, predictions, dates=None)
```

**'evaluate_model'**

The 'evaluate_model' function is responsible for evaluating the performance of a trained LSTM model, providing various metrics such as RMSE, KGE, MAE, and R^2. It also visualises the difference between true and predicted values.

How to use:
1. Import the function
```bash
from tool import evaluate_model
```

2. Evaluate model
- test_loader: DataLoader containing test data.
- model: Trained LSTM model.
- dataset: A 'sweDataset' object with methods to reverse the scaling.
```bash
rmse_test, mae_test, mbe_test, r2_test, kge_test, y_test_ori, test_pred = evaluate_model(test_loader, model, dataset)
```

### Documentation

The tool `html` directory contains detailed documentation for the package, including examples and details of all functions and inputs.

This can be viewed in your browser through the `index.html` file, which is in docs/_build/html/index.html.

### Testing

The tool includes several tests to validate its operations on your system. Before executing these tests, ensure you have [pytest](https://doc.pytest.org/en/latest) installed.

To run the tests, first navigate to the 'swe_tool' directory by using the following command:

```bash
cd swe_tool
```

then execute the following command to run all test files within the directory:

```bash
pytest tests
```

If you want to run one of the specific test files, run the following command:

```bash
pytest tests/<test_filename>
```

### Authors

- Yulin Zhuo

With thanks to Niamh French, Philippa Mason & Corinna Frank for their support.


### References
[1] Siirila-Woodburn, Erica R., Alan M. Rhoades, Benjamin J. Hatchett, Laurie S. Huning, Julia Szinai, Christina Tague, Peter S. Nico, et al. 2021. ‘A Low-to-No Snow Future and Its Impacts on Water Resources in the Western United States’. Nature Reviews Earth & Environment 2 (11): 800–819. https://doi.org/10.1038/s43017-021-00219-y.

[2]Barnett, T. P., J. C. Adam, and D. P. Lettenmaier. 2005. ‘Potential Impacts of a Warming Climate on Water Availability in Snow-Dominated Regions’. Nature 438 (7066): 303–9. https://doi.org/10.1038/nature04141.
### License

[MIT](https://choosealicense.com/licenses/mit/)
