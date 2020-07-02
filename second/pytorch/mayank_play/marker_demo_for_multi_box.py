#!/usr/bin/env python
import math

import rospy
import time
from visualization_msgs.msg import Marker,MarkerArray
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
import geometry_msgs.msg as geom_msg
import std_msgs.msg as std_msg
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2

def wait_for_time():
    """Wait for simulated time to begin.
    """
    while rospy.Time().now().to_sec() == 0:
        pass

def main():
  rospy.init_node('my_node')

  def show_text_in_rviz_mullti_cube(marker_publisher, text):
      markers_my = MarkerArray()
      markers_my.markers = []
      for i in range(5):
          print(time.time())
          ###################################################333333
          marker = Marker(
              type=Marker.CUBE,
              lifetime=rospy.Duration(5),
              pose=Pose(Point(0.5+(i*20), 0.5+(i*20), 1.45), Quaternion(i,i,i, 1)),
              scale=Vector3(0.6, 0.6, 0.6),
              header=Header(frame_id='base_link'),
              color=ColorRGBA(0.0, 1, 0.0, .2))
          marker.action = Marker.ADD
          marker.ns = "est_pose_" + str(i)
          marker.id = 42 + i
          marker.header.stamp = marker.header.stamp
          marker.pose.orientation.w = 1.0
          marker.pose.position.x = (0.5+(i*2))
          marker.pose.position.y =(0.5+(i*2))
          marker.pose.position.z = 1.45

          print(i)
          # rospy.sleep(1)
          ###############################################################
          # marker = Marker(type=Marker.CUBE, ns='velodyne', action=Marker.ADD)
          # marker.header.frame_id = "velodyne"
          # marker.header.stamp = rospy.Time.now()
          # # if self.bbox_data[i][0][0] == frame:
          # # marker.type=Marker.CUBE
          # # marker.scale.x = 0.02
          # marker.scale.x = 0.45+i
          # marker.scale.y = 0.45+i
          # marker.scale.z = 0.45+i
          # marker.lifetime = rospy.Duration.from_sec(5)
          # marker.color.a =.6
          # marker.color.r = 0
          # marker.color.g = 1
          # marker.color.b = 0
          #####################################################################
          markers_my.markers.append(marker)
          # rospy.sleep(1)
      marker_publisher.publish(markers_my)

  def show_text_in_rviz_mullti(marker_publisher, text):
      markers = MarkerArray()
      for i in range(3):
          marker = Marker(type=Marker.LINE_LIST, ns='velodyne', action=Marker.ADD)
          marker.header.frame_id = "velodyne"
          marker.header.stamp = rospy.Time.now()
          # if self.bbox_data[i][0][0] == frame:

          for n in range(8):
              point = geom_msg.Point(self.bbox_data[i][n + 1][0], self.bbox_data[i][n + 1][1], self.bbox_data[i][n + 1][1])
              marker.points.append(point)

          marker.scale.x = 0.02
          marker.lifetime = rospy.Duration.from_sec(0.1)
          marker.color.a = 1.0
          marker.color.r = 0.5
          marker.color.g = 0.5
          marker.color.b = 0.5
          rospy.sleep(1)
          markers.markers.append(marker)

      marker_publisher.publish(markers)

  def show_text_in_rviz_mayank(marker_publisher, text):
      marker = Marker(
          type=Marker.CUBE,
          id=0,
          lifetime=rospy.Duration(5),
          pose=Pose(Point(1, 6,6), Quaternion(0, 0, 0, 1)),
          scale=Vector3(.6, .6, .6),
          header=Header(frame_id='base_link'),
          color=ColorRGBA(0.0, 1.0, 0.0, 0.8),
          text=text)
      marker_publisher.publish(marker)
      print("working")
  # wait_for_time()



  def show_text_in_rviz(marker_publisher, text):
      marker = Marker(
          type=Marker.TEXT_VIEW_FACING,
          id=0,
          lifetime=rospy.Duration(5),
          pose=Pose(Point(0.5, 0.5, 1.45), Quaternion(0, 0, 0, 1)),
          scale=Vector3(0.06, 0.06, 0.06),
          header=Header(frame_id='base_link'),
          color=ColorRGBA(0.0, 1.0, 0.0, 0.8),
          text=text)
      marker_publisher.publish(marker)
      print("working")




  marker_publisher = rospy.Publisher('_mayank_visualization_marker', MarkerArray, queue_size=5)

  for val in range(1000):
        rospy.sleep(1)
        show_text_in_rviz_mullti_cube(marker_publisher, 'Hello world!')
        print("2nd step")

if __name__ == '__main__':
  main()

