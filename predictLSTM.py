import torch
import models
import numpy as np

class predictLSTM():
    def __init__(self,model_path='/Users/yz6622/Desktop/IRP/models/global_model.pth'):
        self.country_models = {}
        self.snowclass_models = {}
        self.general_model = load_model(model_path=model_path, type='general')

    def add_country_model(self, country, model_path):
        model = load_model(model_path, type='countries', sub_type=country)
        self.country_models[country] = model

    def add_snowclass_model(self, snow_class, model_path):
        model = load_model(model_path, type='snowclass', sub_type=snow_class)
        self.snowclass_models[snow_class] = model

    def predict_by_country(self, country, input_data):
        # make sure the input data is torch tensor
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data)
        # make sure the input data is double
        input_data = input_data.float()

        model = self.country_models.get(country)
        if model:
            return model(input_data)
        else:
            raise ValueError(f"No model found for country: {country}")

    def predict_by_snowclass(self, snow_class, input_data):
        # make sure the input data is torch tensor
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data)
        # make sure the input data is double
        input_data = input_data.float()
        
        model = self.snowclass_models.get(snow_class)
        if model:
            return model(input_data)
        else:
            raise ValueError(f"No model found for snow class: {snow_class}")
        
    def predict_by_general(self, input_data):
        return self.general_model(input_data)


def model_class(type='general', sub_type=None):
    """
    Return a specific LSTM model based on the provided type and sub_type.

    :param type: The general category of the model, can be 'general', 'countries', or 'snowclass'. Default is 'general'.
    :type type: str, optional
    :param sub_type: Further specific category based on the 'type'. For 'countries', it should be a country name; for 'snowclass', it should be a number.
    :type sub_type: str or int, optional

    :return: A LSTM model.
    :rtype: torch.nn.Module
    """
    # The general model
    if type == 'general':
        base_model = models.LSTM(input_dim=1, hidden_dims=[60, 30], num_epochs=100) 
    # The models for ensemble model based on countries
    elif type == 'countries':
        if sub_type == 'canada':
            base_model = models.LSTM(input_dim=1, hidden_dims=[60, 30], num_epochs=60)
        elif sub_type == 'switzerland':
            base_model = models.LSTM(input_dim=1, hidden_dims=[70, 50, 30], num_epochs=100)
        elif sub_type == 'norway':
            base_model = models.LSTM(input_dim=2, hidden_dims=[70, 40, 20], num_epochs=100)
        else:
            raise ValueError("'sub_type' should either be 'canada', 'switzerland' or 'norway'.")
    # The models for ensemble model based on snow class
    elif type == 'snowclass':
        if sub_type == 1:
            base_model = models.LSTM(input_dim=1, hidden_dims=[70, 50, 30], num_epochs=100)
        elif sub_type == 3:
            base_model = models.LSTM(input_dim=1, hidden_dims=[70, 50, 30], num_epochs=100)
        elif sub_type == 6:
            base_model = models.LSTM(input_dim=1, hidden_dims=[70, 50, 30], num_epochs=100)
        else:
            raise ValueError("'type' should be an integer from 1-7.")
    else:
        raise ValueError("'type' should either be 'general', 'countries' or 'snowclass'.")

    return base_model   


def load_model(model_path, type='general', sub_type=None):
    """
    Load a certain type of pre-trained LSTM model from the given path.

    :param model_path: The path where the pre-trained model is saved.
    :type model_path: str
    :param type: The general category of the model, can be 'general', 'countries', or 'snowclass'. Default is 'general'.
    :type type: str, optional
    :param sub_type: Further specific category of the model, can be 'general', 'countries', or 'snowclass'. Default is 'general'.
    :type sub_type: str, optional

    :return: A LSTM model.
    :rtype: torch.nn.Module
    """
    model = model_class(type, sub_type)
    model.load_state_dict(torch.load(model_path))
    model = model.float()
    return model


def add_snowclass_model(snow_class, model_path):
    snowclass_models = {}
    model = load_model(model_path, type='snowclass', sub_type=snow_class)
    snowclass_models[snow_class] = model
    return snowclass_models