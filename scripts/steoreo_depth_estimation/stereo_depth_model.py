import numpy as np
import torch 
from utils.parameters import get_nested_item
from tqdm import tqdm

class StereoDepthModel:
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
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()
    self.model.eval()
    raise NotImplementedError    

  def predict(self, left_image: np.ndarray, right_image: np.ndarray)-> np.ndarray:
    """
    Predict depth from image.

    @param left_image: input image, np.ndarray
    @param right_image: input image, np.ndarray
    @return: predicted depth, np.ndarray

    """

     

  






