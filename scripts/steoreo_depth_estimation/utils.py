def load_calibrations(calib_file):
    """
    Loads the calibration file into a dictionary
    """
    calib = {}
    with open(calib_file, 'r') as f:
        for line in f.readlines():
            # each line is of the form 'key=val'
            key, val = line.strip().split('=', 1)
            # val could be a list or a single value 
            try:
                val = float(val)
            except ValueError:
                val = np.array([float(x) for x in re.split('[\[\],;]', val) if x])
            calib[key] = val
        # add fx, fy, cx1, cy, cx2 into the dictionary
        calib['fx'] = calib['cam0'][0]
        calib['fy'] = calib['cam0'][2]
        calib['cx1'] = calib['cam0'][4]
        calib['cy'] = calib['cam0'][5]
        calib['cx2'] = calib['cam1'][4]
    return calib


def disparity_to_depth(flow_up, calib):
    """Converts a disparity map to a depth map"""
    depth = calib['fx'] * calib['baseline'] / np.abs(-flow_up + (calib['cx2'] - calib['cx1']))
    return depth