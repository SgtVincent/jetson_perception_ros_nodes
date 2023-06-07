#!/usr/bin/env python
"""
The node mono_depth_node keeps subscribing to a ROS Image topic and publishing the predicted depth from models.
"""
import os
from os.path import dirname, abspath
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import yaml 

from monocular_depth_estimation.mono_depth_model import create_model
from utils.parameters import get_nested_item


class MonoDepthNode:
    def __init__(self):
        self._read_params()
        self._init_from_params()
        

    def _read_params(self):
        """
        Read parameters from ROS parameter server.
        """
        # TODO: rosparam load yaml in roslaunch does not work properly, just load parameters from yaml
        # self.params = rospy.get_param('/monocular_depth_estimation')  
        config_path = os.path.join(dirname(dirname(os.path.abspath(__file__))), 'config/default.yaml')
        with open(config_path, 'r') as f:
            self.params = get_nested_item(yaml.safe_load(f), 'monocular_depth_estimation')
        

        self.enabled = get_nested_item(self.params, 'enabled')
        self.source_topic = get_nested_item(self.params, 'source_topic')
        self.target_topic = get_nested_item(self.params, 'target_topic')
        self.model_params = get_nested_item(self.params, 'model')

    def _init_from_params(self):
        """
        Initialize member variables from ROS parameters.
        """
        if not self.enabled:
            rospy.loginfo("Mono depth estimation is disabled.")
            return
        # initialize model from model_params
        self.model = create_model(self.model_params)

        # initialize ROS subscribers and publishers
        self.bridge = CvBridge()
        # always get latest image from source_topic
        self.image_sub = rospy.Subscriber(self.source_topic, CompressedImage, self.image_callback, queue_size=1)
        self.depth_pub = rospy.Publisher(self.target_topic, Image, queue_size=1)
        # self.depth_compressed_pub = rospy.Publisher(self.target_topic, CompressedImage, queue_size=1)
        

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)
            return

        # Perform depth prediction using your models
        predicted_depth = self.model.predict(cv_image)

        # Convert predicted depth to ROS Image message
        # depth_msg = self.bridge.cv2_to_imgmsg(predicted_depth, "32FC1")
        depth_msg = self.bridge.cv2_to_imgmsg(predicted_depth, "passthrough")
        # depth_msg = self.bridge.cv2_to_compressed_imgmsg(predicted_depth)

        # Publish the predicted depth
        self.depth_pub.publish(depth_msg)
        # self.depth_compressed_pub.publish(depth_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':

    rospy.init_node('~')

    depth_prediction_node = MonoDepthNode()
    
    if depth_prediction_node.enabled:
        depth_prediction_node.run()