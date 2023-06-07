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

    self._load_model()

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


# models depending on the jetson-inference library
try:

  # deprecated   
  # import jetson.inference
  # import jetson.utils

  import jetson_inference
  import jetson_utils

  class FastDepthNet(MonoDepthModel):
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

    def _load_model(self):
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
      depth_net_params = get_nested_item(self.params, 'FastDepthNet')
      
      # Not using model_path for now, only use built-in cached models
      self.model = jetson_inference.depthNet(**depth_net_params)
      
      # prepare for the memory mapping 
      self.depth_field = self.model.GetDepthField()
      self.depth_filed_numpy = jetson_utils.cudaToNumpy(self.depth_field)


    def predict(self, image: np.ndarray)-> np.ndarray:
      """
      Predict depth from image.
      """
      # convert numpy array to jetson_utils.cudaImage
      cuda_image = jetson_utils.cudaFromNumpy(image)
      # Predict depth from image
      self.model.Process(cuda_image)
      # wait for GPU to finish processing, so we can use the results on CPU
      jetson_utils.cudaDeviceSynchronize() 
      # copy the memory mapping to numpy array

      # TODO: comment out the following line to disable printing FPS
      rospy.loginfo("{:s} Network {:.1f} FPS".format(self.model.GetNetworkName(), self.model.GetNetworkFPS()))

      return self.depth_filed_numpy.copy()
  
except ImportError:
  rospy.logwarn("jetson-inference library not found, FastDepth model will not be available.")



def create_model(model_params: Dict[str, Any]) -> 'MonoDepthModel':
  """
  Create a model from model_params.
  """
  model_name = model_params['model_name']
  # get the class object from the model name and create an instance from the params
  return globals()[model_name](model_params)