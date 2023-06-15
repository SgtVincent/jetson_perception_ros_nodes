import numpy as np

from utils.parameters import get_nested_item

class MonoDepthModel:
  """
  This class provides a unified interface for loading and using different monocular depth estimation models.
  """

  def __init__(self, params):
    """
    Initialize the model.
    """
    self.params = params
    self.model_name = params['model_name']
    self.model_params = get_nested_item(self.params, self.model_name)

  def _load_model(self):
    """
    Load the model from model_path.
    """
    raise NotImplementedError    

  def predict(self, image: np.ndarray)-> np.ndarray:
    """
    Predict depth from image.

    @param image: input image, np.ndarray
    @return: predicted depth, np.ndarray

    """
    raise NotImplementedError


  






