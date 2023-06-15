from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
import numpy as np
import torch

from utils.parameters import get_nested_item
from .mono_depth_model import MonoDepthModel


class GLPN(MonoDepthModel):
  """
  The wrapper class for Global-Local Path Networks https://arxiv.org/abs/2201.07436

  The implementation is integrated by hugging face transformer library, see more details here: 
  https://huggingface.co/vinvino02/glpn-kitti
  """

  def __init__(self, params):
    """
    Initialize the model.
    """
    super().__init__(params)
    self.network = get_nested_item(self.model_params, 'network')
    self.model_cache_dir = get_nested_item(self.model_params, 'model_cache_dir')

    self._load_model()


  def _load_model(self):
    """
    Load the model from model path.
    """
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GLPNFeatureExtractor is simply an image pre-processing pipeline, cannot be ported to GPU
    self.feature_extractor = GLPNFeatureExtractor.from_pretrained(
      self.network, cache_dir=self.model_cache_dir)
    # Load the model and move it to GPU
    self.model = GLPNForDepthEstimation.from_pretrained(
      self.network, cache_dir=self.model_cache_dir).to(self.device)
    

  def predict(self, image: np.ndarray) -> np.ndarray:
    """
    Predict depth from image.

    @param image: input image, np.ndarray
    @return: predicted depth, np.ndarray

    """
    # pre-process the image to desired features
    inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)

    with torch.no_grad():
        outputs = self.model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    interpolated_depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = interpolated_depth.squeeze().cpu().numpy()
    return output
