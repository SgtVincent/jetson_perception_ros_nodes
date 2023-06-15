from typing import Any, Dict

from .mono_depth_model import MonoDepthModel



def create_model(model_params: Dict[str, Any]) -> 'MonoDepthModel':
  """
  Create a model from model_params.
  """
  model_name = model_params['model_name']

  ####################################################################
  # Actively import models here to avoid import conflicts 
  ####################################################################

  if model_name == 'FastDepth':

    from .fast_depth import FastDepth
    return FastDepth(model_params)

  elif model_name == 'GuidedDecoding':

    from .guide_depth import GuidedDecoding
    return GuidedDecoding(model_params)

  elif model_name == 'GLPN':
    
    from .glpn import GLPN
    return GLPN(model_params)

  else:
    raise ValueError("Unsupported model name: {:s}".format(model_name))
  
  