from typing import Any, Dict

from .raft_stereo import RaftStereoModel



def create_model(model_params: Dict[str, Any]):
  """
  Create a model from model_params.
  """
  model_name = model_params['model_name']

  ####################################################################
  # Actively import models here to avoid import conflicts 
  ####################################################################
  if model_name == 'raft_stereo':
    return RaftStereoModel(model_params)
 

  else:
    raise ValueError("Unsupported model name: {:s}".format(model_name))
  
  