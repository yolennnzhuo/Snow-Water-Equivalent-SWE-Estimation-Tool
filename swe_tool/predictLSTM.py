# Name: Yulin Zhuo 
# Github username: edsml-yz6622

import torch
from models import LSTM
import numpy as np

class predictLSTM():
    """
    A class designed to use a pre-trained LSTM Model to predict SWE value.

    Attributes:
        - country_models: A dictionary where store the LSTM models based on
          countries.
        - snowclass_models: A dictionary where store the LSTM models based 
          on snow classes.
        - type: The general category of the model.
        - sub_type: The further specific category based on the 'type'.
        - model: The pre-trained LSTM model.

    Methods:
        - __call__: Make the instance to use the appropriate prediction 
          method.
        - add_country_model(x): Ensemble all the LSTM models based on 
          countries by adding to dictionary.
        - add_snowclass_model(x): Ensemble all the LSTM models based 
          on different snow classes by adding to dictionary.
        - predict_by_country(x): Apply the ensemble LSTM models with a
          specific location to predict the SWE.
        - predict_by_snowclass(x): Apply the ensemble LSTM models with
          a specific snow class to predict the SWE.
        - predict_by_general(x): Apply the general LSTM models to predict
          the SWE (The model is trained on the whole dataset - across countries).
    """
    def __init__(self,model_path='../models/global_model.pth',type='general'
                 ,sub_type=None):
        """
        Initialise the predictLSTM class.

        :param model_path: The path stored the pre-trained LSTM model, 
                           default is '../models/global_model.pth'.
        :type model_path: str, optional
        :param type: The general category of the model, can be 'general', 
                    'countries', or 'snowclass'. Default is 'general'.
        :type type: str, optional
        :param sub_type: Further specific category based on the 'type'. 
                         For 'countries', it should be a country name;
                         for 'snowclass', it should be a number. 
                         Default is None.
        :type sub_type: str or int, optional
        """
        self.country_models = {}
        self.snowclass_models = {}
        self.type = type
        self.sub_type = sub_type
        self.model = load_model(model_path=model_path, type=type, sub_type=sub_type)
    
    def __call__(self, input_data):
        """
        Call the appropriate model based on type and sub_type for prediction.

        :param input_data: The input data requires for the models.
        :type input_data: list or numpy.array

        :return: The predicted value of SWE.
        :rtype: torch.Tensor
        """
        if self.type == 'general':
            return self.predict_by_general(input_data)
        elif self.type == 'countries':
            return self.predict_by_country(self.sub_type, input_data)
        elif self.type == 'snowclass':
            return self.predict_by_snowclass(self.sub_type, input_data)
        else:
            raise ValueError(f"Unsupported type: {self.type}")
        
    def input_preprocess(self, input_data):
        """
        Convert the input data to torch tensor and float

        :param input_data: The input data requires for the models.
        :type input_data: list or numpy.array

        :return: The processed input data.
        :rtype: torch.Tensor
        """
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data)
        elif isinstance(input_data, list):
            input_data = torch.tensor(input_data)

        return input_data.float()

    def add_country_model(self, country, model_path):
        """
        Ensemble all the pre-trained LSTM model (Add all the models 
        to 'country_models' dictionary)

        :param country: The name of country where needs to predict SWE.
        :type country: str
        :param model_path: The path stored the pre-trained LSTM model.
        :type model_path: str
        """
        model = load_model(model_path, type='countries', sub_type=country)
        self.country_models[country] = model

    def add_snowclass_model(self, snow_class, model_path):
        """
        Ensemble all the pre-trained LSTM model (Add all the models to 
        'snowclass_models' dictionary)

        :param snow_class: The snow class category, either 1, 2, 3, 5, or 6.
        :type snow_class: int
        :param model_path: The path stored the pre-trained LSTM model.
        :type model_path: str
        """
        model = load_model(model_path, type='snowclass', sub_type=snow_class)
        self.snowclass_models[snow_class] = model

    def predict_by_country(self, country, input_data):
        """
        Call the ensemble models to predict the SWE value by passing the 
        required input data(X).

        :param country: The name of country where needs to predict SWE.
        :type country: str
        :param input_data: The input data requires for the models.
        :type input_data: list or numpy.array

        :return: model(input_data): The predicted value of SWE.
        :rtype: torch.Tensor
        """
        # make sure the input data is torch tensor and float
        input_data = self.input_preprocess(input_data)

        model = self.country_models.get(country)
        if model:
            return model(input_data)
        else:
            raise ValueError(f"No model found for country: {country}")

    def predict_by_snowclass(self, snow_class, input_data):
        """
        Call the ensemble models to predict the SWE value by passing the 
        required input data(X).

        :param snow_class: The snow class category, either 1, 2, 3, 5, or 6.
        :type snow_class: int
        :param input_data: The input data requires for the models.
        :type input_data: list or numpy.array

        :return: model(input_data): The predicted value of SWE.
        :rtype: torch.Tensor
        """
        # make sure the input data is torch tensor and float
        input_data = self.input_preprocess(input_data)

        model = self.snowclass_models.get(snow_class)
        if model:
            return model(input_data)
        else:
            raise ValueError(f"No model found for snow class: {snow_class}")
        
    def predict_by_general(self, input_data):
        """
        Call the general model to predict the SWE value by passing the 
        required input data(X).

        :param input_data: The input data requires for the models.
        :type input_data: list or numpy.array

        :return: model(input_data): The predicted value of SWE.
        :rtype: torch.Tensor
        """
        input_data = self.input_preprocess(input_data)
        return self.model(input_data)


def model_class(type='general', sub_type=None):
    """
    Return a specific LSTM model based on the provided type and sub_type.

    :param type: The general category of the model, can be 'general', 
                 'countries', or 'snowclass'. Default is 'general'.
    :type type: str, optional
    :param sub_type: Further specific category based on the 'type'. 
                     For 'countries', it should be a country name; 
                     For 'snowclass', it should be a number.
    :type sub_type: str or int, optional
    :return: A LSTM model.
    :rtype: torch.nn.Module
    """
    # The general model
    if type == 'general':
        base_model = LSTM(input_dim=1, hidden_dims=[60, 30], num_epochs=100) 
    # The models for ensemble model based on countries
    elif type == 'countries':
        if sub_type == 'canada':
            base_model = LSTM(input_dim=1, hidden_dims=[60, 30], num_epochs=60)
        elif sub_type == 'switzerland':
            base_model = LSTM(input_dim=1, hidden_dims=[70, 50, 30], num_epochs=100)
        elif sub_type == 'norway':
            base_model = LSTM(input_dim=2, hidden_dims=[70, 40, 20], num_epochs=100)
        elif sub_type == 'US':
            base_model = LSTM(input_dim=1, hidden_dims=[60, 30], num_epochs=100)
        else:
            raise ValueError("'sub_type' should either be 'US', 'canada','switzerland' or 'norway'.")
    # The models for ensemble model based on snow class
    elif type == 'snowclass':
        if sub_type == 1:
            base_model = LSTM(input_dim=1, hidden_dims=[70, 50, 30], num_epochs=60)
        elif sub_type == 2:
            base_model = LSTM(input_dim=1, hidden_dims=[60, 30], num_epochs=30)
        elif sub_type == 3:
            base_model = LSTM(input_dim=3, hidden_dims=[70, 50, 30], num_epochs=200)
        elif sub_type == 5:
            base_model = LSTM(input_dim=1, hidden_dims=[60, 30], num_epochs=50)
        elif sub_type == 6:
            base_model = LSTM(input_dim=1, hidden_dims=[50], num_epochs=100)
        else:
            raise ValueError("'type' should be an integer from 1,2,3,5,6.")
    else:
        raise ValueError("'type' should either be 'general', 'countries' or 'snowclass'.")

    return base_model   


def load_model(model_path, type='general', sub_type=None):
    """
    Load a certain type of pre-trained LSTM model from the given path.

    :param model_path: The path where the pre-trained model is saved.
    :type model_path: str
    :param type: The general category of the model, can be 'general', 'countries', 
                 or 'snowclass'. Default is 'general'.
    :type type: str, optional
    :param sub_type: Further specific category based on the 'type'. 
                     For 'countries', it should be a country name; 
                     for 'snowclass', it should be a number.
    :type sub_type: str, optional

    :return: A LSTM model.
    :rtype: torch.nn.Module
    """
    model = model_class(type, sub_type)

    try:
        model.load_state_dict(torch.load(model_path))
        
    # Error handling
    except FileNotFoundError:
        raise ValueError(f"Model file not found at : {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model from : {model_path}. Error messages : {e}")
    
    model = model.float()
    return model