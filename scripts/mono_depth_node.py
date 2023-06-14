#!/usr/bin/env python
"""
The node mono_depth_node keeps subscribing to a ROS Image topic and publishing the predicted depth from models.
"""
import os
from os.path import dirname, abspath
import rospy
import numpy as np
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
        self.input_compressed = get_nested_item(self.params, 'input_compressed')
        self.output_compressed = get_nested_item(self.params, 'output_compressed')
        self.process_rate = get_nested_item(self.params, 'process_rate')
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
        if self.input_compressed:
            self.image_sub = rospy.Subscriber(self.source_topic, CompressedImage, self.image_callback, queue_size=1)
        else:
            self.image_sub = rospy.Subscriber(self.source_topic, Image, self.image_callback, queue_size=1)

        if self.output_compressed:
            self.depth_pub = rospy.Publisher(self.target_topic, CompressedImage, queue_size=1)
        else:
            self.depth_pub = rospy.Publisher(self.target_topic, Image, queue_size=1)

        self.last_process_time = rospy.Time(0, 0)

    def image_callback(self, msg):

        # Only process the image if the time interval is greater than the process rate
        current_time = rospy.Time.now()
        if (current_time - self.last_process_time).to_sec() < 1.0 / self.process_rate:
            return
        self.last_process_time = current_time

        # Convert ROS Image message to cv2 image
        if self.input_compressed:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
        else:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

        # Perform depth prediction using your models
        predicted_depth = self.model.predict(cv_image).astype(np.float32)

        # Convert predicted depth to ROS Image message
        if self.output_compressed:
            depth_msg = self.bridge.cv2_to_compressed_imgmsg(predicted_depth)
        else:
            depth_msg = self.bridge.cv2_to_imgmsg(predicted_depth, "32FC1")

        # Publish the predicted depth
        self.depth_pub.publish(depth_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':

    rospy.init_node('~')

    depth_prediction_node = MonoDepthNode()
    
    if depth_prediction_node.enabled:
        depth_prediction_node.run()