#!/usr/bin/env python
"""
The node mono_depth_node keeps subscribing to a ROS Image topic and publishing the predicted depth from models.
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

from monocular_depth_estimation.mono_depth_model import MonoDepthModel
from utils.parameters import get_nested_item


class MonoDepthNode:
    def __init__(self):
        self._read_params()
        self._init_from_params()
        

    def _read_params(self):
        """
        Read parameters from ROS parameter server.
        """
        self.params = rospy.get_param('/monocular_depth_estimation')
        self.enabled = get_nested_item(self.params, 'enabled')
        self.source_topic = rospy.get_param('source_topic')
        self.target_topic = rospy.get_param('target_topic')
        self.model_params = rospy.get_param('model')

    def _init_from_params(self):
        """
        Initialize member variables from ROS parameters.
        """
        if not self.enabled:
            rospy.loginfo("Mono depth estimation is disabled.")
            return
        # initialize model from model_params
        self.model = MonoDepthModel.create_model(self.model_params)


    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)
            return

        # Perform depth prediction using your models
        predicted_depth = self.predict_depth(cv_image)

        # Convert predicted depth to ROS Image message
        depth_msg = self.bridge.cv2_to_imgmsg(predicted_depth, "32FC1")

        # Publish the predicted depth
        self.depth_pub.publish(depth_msg)

    def predict_depth(self, image):
        # Implement your depth prediction using models
        # This is just a placeholder, replace it with your actual code
        predicted_depth = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return predicted_depth

    def run(self):
        rospy.spin()

if __name__ == '__main__':

    rospy.init_node('~')

    depth_prediction_node = MonoDepthNode()
    
    if depth_prediction_node.enabled:
        depth_prediction_node.run()