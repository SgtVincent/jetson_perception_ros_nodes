import numpy as np
import sys
import torch 

from utils.parameters import get_nested_item
sys.path.append(join(dirname(abspath(__file__)), 'GuidedDecoding'))
from model.loader import load_model as gd_load_model


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
    with torch.no_grad():

      padder = InputPadder(image1.shape, divis_by=32)
      image1, image2 = padder.pad(image1, image2)

      _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
      flow_up = padder.unpad(flow_up).squeeze()

      file_stem = imfile1.split('/')[-2]
      if args.save_numpy:
          np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
      if args.save_depth:
          calib_file = os.path.join(os.path.dirname(imfile1), 'calib.txt')
          calib = load_calibrations(calib_file)
          depth = disparity_to_depth(flow_up.cpu().numpy().squeeze(), calib)
          plt.imsave(output_directory / f"{file_stem}_depth.png", depth)
              


  






