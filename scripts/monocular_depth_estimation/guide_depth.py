import sys
from os.path import abspath, dirname, join
import torch 
import numpy as np
import torch
import torchvision

from utils.parameters import get_nested_item
from .mono_depth_model import MonoDepthModel

# add the GuidedDecoding repo to the python path
sys.path.append(join(dirname(abspath(__file__)), 'GuidedDecoding'))
from model.loader import load_model as gd_load_model

class GuidedDecoding(MonoDepthModel):
  """
  This class provides a unified interface for loading and using GuidedDecoding model.
  For more information about GuidedDecoding, see:
  https://github.com/mic-rud/guideddecoding
  """

  def __init__(self, params):
    """
    Initialize the model.
    """
    super().__init__(params)
    self.network = get_nested_item(self.model_params, 'network')
    self.model_path = get_nested_item(self.model_params, 'model_path')
    self.resolution = get_nested_item(self.model_params, 'resolution')
    self.max_depth = get_nested_item(self.model_params, 'max_depth')

    # load model from model_path
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self._load_model()

    # pre-processing transforms
    self.downscale_image = torchvision.transforms.Resize(self.resolution) # To model resolution
    self.to_tensor = torchvision.transforms.ToTensor() # To Tensor

  def predict(self, image: np.ndarray)-> np.ndarray:
    """ 
    Predict depth from image.
    """
    with torch.no_grad():
      # pre-process the image 
      image_tensor = self._preprocess(image)

      inv_depth = self.model(image_tensor)
      depth = self._inverse_depth_norm(inv_depth)
      return depth.squeeze().detach().cpu().numpy()

  def _load_model(self):
    """
    Load the model from model_path.
    """
    self.model = gd_load_model(self.network, self.model_path)
    self.model.to(self.device)
    self.model.eval()

  def _preprocess(self, image: np.ndarray)-> np.ndarray:
    """
    Preprocess the input RGB image.
    """
    # normalize image to [0, 1]
    normalized_image = image.astype(np.float32) / 255.0

    # convert numpy array to torch tensor and add batch dimension
    image_tensor = self.to_tensor(normalized_image).to(self.device, non_blocking=True)
    image_tensor = image_tensor.unsqueeze(0)

    # rescale the image to model resolution
    image_tensor = self.downscale_image(image_tensor)

    return image_tensor

  def _inverse_depth_norm(self, depth):
    """
    Inverse depth normalization.
    """
    depth = self.max_depth / depth
    # TODO: consider change min depth to 0.0? 
    depth = torch.clamp(depth, self.max_depth / 100, self.max_depth)
    return depth
