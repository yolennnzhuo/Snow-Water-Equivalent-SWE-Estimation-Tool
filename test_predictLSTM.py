import pytest
import torch
import numpy as np
from predictLSTM import predictLSTM, model_class, load_model
from models import LSTM

class TestPredictLSTM():
    path = '/Users/yz6622/Desktop/IRP/models/'

    @pytest.fixture(autouse=True)
    def setUp(self):
        """
        Setup test data and initialise the instance.
        """
        self.model_general = predictLSTM()
        self.input_array = np.array([[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]]) # (10,1,1)
        self.input_list = [[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]] # (10,1,1)

    def test_init(self):
        """
        Test the initialisation of predictLSTM.
        """
        assert isinstance(self.model_general.country_models, dict), "country_models should be dic."
        assert isinstance(self.model_general.snowclass_models, dict), "snowclass_models should be dic."
        assert isinstance(self.model_general.model, LSTM), "model should be an instance of LSTM."
        assert len(self.model_general.country_models) == 0, "country_models NOT empty."
        assert len(self.model_general.snowclass_models) == 0, "snowclass_models NOT empty."

    def test_country_model_init(self):
        """
        Test the initialisation of the country model in predictLSTM.
        """
        models_country = predictLSTM(model_path= self.path +'model_norway.pth'
                                     , type='countries', sub_type='norway')
        assert isinstance(models_country.model, LSTM), "Expected country model to be an instance of LSTM."

    def test_snowclass_model_init(self):
        """
        Test the initialisation of the snow class model in predictLSTM.
        """
        models_sc = predictLSTM(model_path=self.path +'model_sc_1.pth', type='snowclass', sub_type=1)
        assert isinstance(models_sc.model, LSTM), "Expected snowclass model to be an instance of LSTM."

    def test_input_pre(self):
        """
        Test the functionality of input preprocessing in predictLSTM.
        """
        processed_list = self.model_general.input_preprocess(self.input_list)
        assert isinstance(processed_list, torch.Tensor), "Expected output data to be Tensor."
        assert processed_list.dtype == torch.float32, "Expected output data to be float."

        processed_array = self.model_general.input_preprocess(self.input_array)
        assert isinstance(processed_array, torch.Tensor), "Expected output data to be Tensor."
        assert processed_array.dtype == torch.float32, "Expected output data to be float."

    def test_add_country_model(self):
        """
        Test to add a new country model to the country_models dictionary.
        """
        self.model_general.add_country_model('US', self.path+'model_US.pth')
        assert 'US' in self.model_general.country_models, "Unexpected key in country_models dictionary."
    
    def test_add_snowclass_model(self):
        """
        Test to add a new snow class model to the snowclass_models dictionary.
        """
        self.model_general.add_snowclass_model(2, self.path+'model_sc_2.pth')
        assert 2 in self.model_general.snowclass_models, "Unexpected key in snowclass_models dictionary."
    
    def test_predict_by_country(self):
        """
        Test to predict for a specific country using predictLSTM.
        """
        self.model_general.add_country_model('canada', self.path+'model_canada.pth')
        output = self.model_general.predict_by_country('canada', self.input_array)
        assert isinstance(output, torch.Tensor), "Unexpected output type."

    def test_predict_by_snowclass(self):
        """
        Test to predict for a specific snow class using predictLSTM.
        """
        self.model_general.add_snowclass_model(5, self.path+'model_sc_5.pth')
        output = self.model_general.predict_by_snowclass(5, self.input_list)
        assert isinstance(output, torch.Tensor), "Unexpected output type."

    def test_model_class(self):
        """
        Test the model_class() for initialising a model instance.
        """
        model = model_class()
        assert isinstance(model, LSTM), "Unexpected model type."

    def test_load_model(self):
        """
        Test the load_model() for loading model correcetly.
        """
        model = load_model(TestPredictLSTM.path + 'global_model.pth')
        assert isinstance(model, LSTM), "Unexpected model type."


    

