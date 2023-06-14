import os 
from os.path import join, dirname, abspath
from typing import Any, Dict
import numpy as np
import sys
import rospy
import torch 
import torchvision
import cv2

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


def create_model(model_params: Dict[str, Any]) -> 'MonoDepthModel':
  """
  Create a model from model_params.
  """
  model_name = model_params['model_name']

  ####################################################################
  # Actively import models here to avoid import conflicts 
  ####################################################################

  if model_name == 'FastDepth':
    # deprecated   
    # import jetson.inference
    # import jetson.utils

    import jetson_inference
    import jetson_utils

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

        self._load_model()

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
        depth_net_params = get_nested_item(self.params, 'FastDepth')
        
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
    
    return FastDepth(model_params)

  elif model_name == 'GuidedDecoding':
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

    return GuidedDecoding(model_params)

  elif model_name == 'GLPN':
    raise NotImplementedError
  
  else:
    raise ValueError("Unsupported model name: {:s}".format(model_name))
  





