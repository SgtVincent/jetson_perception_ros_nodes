def get_nested_item(parameters: dict, name, default=None):
  """
  Returns a nested item from a parameter tree.
  """
  split_name = name.split('/')
  for name in split_name:
    try:
      parameters = parameters[name]
    except:
      return default
  return parameters