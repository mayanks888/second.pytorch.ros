#!/usr/bin/env python
# ROS node libs

import rospy
from sensor_msgs.msg import Image,PointCloud2
from std_msgs.msg import Int16, Float32MultiArray

from cv_bridge import CvBridge, CvBridgeError

# General libs
import numpy as np
import os
import sys

import cv2
# import PIL
import time
import time
import datetime


# GPU settings: Select GPUs to use. Coment it to let the system decide
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

class ros_tensorflow_obj():
    def __init__(self):
        # ## Initial msg
        rospy.loginfo('  ## Starting ROS  interface ##')
        # ## Load a (frozen) Tensorflow model into memory.
        print("ready to process----------------------------------------------------------")
        # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
        # ROS environment setup
        # ##  Define subscribers
        self.subscribers_def()
        # print("subs")
        # ## Define publishers
        self.publishers_def()
        # print("pub")
        # ## Get cv_bridge: CvBridge is an object that converts between OpenCV Images and ROS Image messages
        self._cv_bridge = CvBridge()
        self.now = rospy.Time.now()

    # Define subscribers
    def subscribers_def(self):
        print("mydata")
        subs_topic = '/kitti/velo/pointcloud'
        # self._sub = rospy.Subscriber( subs_topic , PointCloud2, self.img_callback, queue_size=1, buff_size=2**24)
        mydata = rospy.Subscriber( subs_topic , PointCloud2, self.img_callback, queue_size=1, buff_size=2**24)

        # self._sub = rospy.Subscriber( subs_topic , Image, self.img_callback, queue_size=1, buff_size=100)

    # Define publishers
    def publishers_def(self):
        tl_bbox_topic = '/tl_bbox_topic_megs'
        self._pub = rospy.Publisher('tl_bbox_topic', Float32MultiArray, queue_size=1)

    # Camera image callback
    def img_callback(self, point_cl_msg):
        print("mydata_call")
        print(point_cl_msg)

        # self._pub.publish(tl_bbox)
# Spin once



def spin(self):
    rospy.spin()

def main():
    rospy.init_node('my_node', anonymous=True)
    tf_ob = ros_tensorflow_obj()
    # tf_ob.subscribers_def
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
if __name__ == '__main__':
    main()