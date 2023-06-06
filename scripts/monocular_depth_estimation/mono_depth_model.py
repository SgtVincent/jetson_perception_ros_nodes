import os 
from typing import Any, Dict
import numpy as np
import sys
import rospy

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
    self.model_path = params['model_path']

    self.load_model()

  def load_model(self):
    """
    Load the model from model_path.
    """
    raise NotImplementedError    

  def predict_depth(self, image: np.ndarray)-> np.ndarray:
    """
    Predict depth from image.
    """
    raise NotImplementedError

  @staticmethod
  def create_model(model_params: Dict[str, Any]):
    """
    Create a model from model_params.
    """
    model_name = model_params['name']
    # get the class object from the model name and create an instance from the params
    return globals()[model_name](model_params)


# models depending on the jetson-inference library
try:
  
  import jetson.inference
  import jetson.utils

  class FastDepth(MonoDepthModel):
    """
    This class provides a unified interface for loading and using FastDepth model.

    For more information about jetson.inference.depthNet, see:
    https://github.com/dusty-nv/jetson-inference/blob/master/docs/depthnet.md

    """

    def __init__(self, params):
      """
      Initialize the model.
      """
      super().__init__(params)

    def load_model(self):
      """
      Load the model from model_path.

      Parameters for jetson.inference.depthNet:
        network (string) -- name of a built-in network to use,
                            see below for available options.

        argv (strings) -- command line arguments passed to depthNet,
                          see below for available options.
      
        depthNet arguments: 
          --network NETWORK    pre-trained model to load, one of the following:
                                  * fcn-mobilenet
                                  * fcn-resnet18
                                  * fcn-resnet50
          --model MODEL        path to custom model to load (onnx)
          --input_blob INPUT   name of the input layer (default is 'input_0')
          --output_blob OUTPUT name of the output layer (default is 'output_0')
          --profile            enable layer profiling in TensorRT
      """
      depth_net_params = get_nested_item(self.params, 'FastDepth')
      
      # Not using model_path for now, only use built-in cached models
      self.model = jetson.inference.depthNet(**depth_net_params)
      
      # prepare for the memory mapping 
      self.depth_field = self.net.GetDepthField()
      self.depth_filed_numpy = jetson.utils.cudaToNumpy(self.depth_field)


    def predict_depth(self, image: np.ndarray)-> np.ndarray:
      """
      Predict depth from image.
      """
      # Predict depth from image
      self.model.Process(image)
      # wait for GPU to finish processing, so we can use the results on CPU
      jetson.utils.cudaDeviceSynchronize() 
      # copy the memory mapping to numpy array
      return self.depth_filed_numpy.copy()
  
except ImportError:
  rospy.logwarn("jetson-inference library not found, FastDepth model will not be available.")