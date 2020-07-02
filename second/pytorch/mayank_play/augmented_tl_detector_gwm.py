#!/usr/bin/env python
# ROS node libs
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image,PointCloud2
from std_msgs.msg import Int16, Float32MultiArray
from multiprocessing import Queue, Pool
from cv_bridge import CvBridge, CvBridgeError

# General libs
import numpy as np
import os
import sys
import tensorflow as tf
import cv2
# import PIL
import time
import time
import datetime
# Detector libs
# from object_detection.utils import ops as utils_ops
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util

# GPU settings: Select GPUs to use. Coment it to let the system decide
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

class ros_tensorflow_obj():
    def __init__(self):
        # ## Initial msg
        rospy.loginfo('  ## Starting ROS Tensorflow interface ##')
        # ## Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            # with tf.gfile.GFile("./object_detection/saved_model_frcnn_4000_samples_00/frozen_inference_graph.pb", 'rb') as fid:
            # with tf.gfile.GFile("./object_detection/mayank_pb/frozen_inference_graph_100000.pb", 'rb') as fid:
            # with tf.gfile.GFile("./object_detection/mayank_pb/frozen_inference_graph_14134.pb", 'rb') as fid:
            with tf.gfile.GFile("./object_detection/mayank_pb/frozen_inference_graph_30812.pb", 'rb') as fid:
            # with tf.gfile.GFile("./object_detection/mayank_pb/frozen_inference_graph_new.pb", 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # ## Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything 			that returns a dictionary mapping integers to appropriate string labels would be fine
        # self.category_index = label_map_util.create_category_index_from_labelmap("./object_detection/Baidu_models/Baidu_ssd_model_0/haval_label_map.pbtxt", use_display_name=True)
        # self.category_index = label_map_util.create_category_index_from_labelmap("./object_detection/mayank_pb/bdd_traffic_label_map_simgle.pbtxt", use_display_name=True)
        # ## Get Tensors to run from model
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # # Tensorflow Session opening: Creates a session with log_device_placement set to True.
        # ## Session configuration
        config=tf.ConfigProto(log_device_placement=True)
        config.log_device_placement = True
        config.gpu_options.allow_growth = True

        # ## Session openning
        try:
            with detection_graph.as_default():
                self.sess = tf.Session(graph=detection_graph, config = config)
                rospy.loginfo('  ## Tensorflow session open: Starting inference... ##')
        except ValueError:
            rospy.logerr('   ## Error when openning session. Please restart the node ##')
            rospy.logerr(ValueError)	

        # image_np= cv2.imread("baid_0.jpg",1)
        # image_np_expanded = np.expand_dims(image_np, axis=0)
        # (boxes_out, scores_out, classes_out, num_detections_out) = self.sess.run(
        # [self.boxes, self.scores, self.classes, self.num_detections], feed_dict={self.image_tensor: image_np_expanded})
#############################################33333
        # loading keras graph
        global mymodel
        mymodel = tf.keras.models.load_model("./saved_models/new_model/model_b_lr-4_ep150_ba32_1.h")
        # this is key : save the graph after loading the model
        global graph
        graph = tf.get_default_graph()
        # self.mymodel = tf.keras.models.load_model("./saved_models/new_model/model_b_lr-4_ep150_ba32.h")
        # print("My model is:", mymodel.summary())
        # self.light_co=[]
        #####################################################
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
        subs_topic = 'crop_tl_image'
        self._sub = rospy.Subscriber( subs_topic , Image, self.img_callback, queue_size=1, buff_size=2**24)
        # self._sub = rospy.Subscriber( subs_topic , Image, self.img_callback, queue_size=1, buff_size=100)

    # Define publishers
    def publishers_def(self):
        tl_bbox_topic = '/tl_bbox_topic_megs'
        self._pub = rospy.Publisher('tl_bbox_topic', Float32MultiArray, queue_size=1)

    # Camera image callback
    def img_callback(self, image_msg):
        image_msg.encoding = "bgr8"
        first = time.time()
        image_np = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        rows, cols, channels = image_np.shape
        image_np_expanded = np.expand_dims(image_np, axis=0)
        second = time.time()
        time_cost = round((second-first)*1000)
        print("post processing Time",time_cost)
        third = time.time()
        (boxes_out, scores_out, classes_out, num_detections_out) = self.sess.run([self.boxes, self.scores, self.classes, self.num_detections],feed_dict={self.image_tensor: image_np_expanded})
        numboxes = np.squeeze(num_detections_out)
        # print("num_detection are",num_detections_out)
        numboxes = 1

        # print("confidence score are",scores_out[:, 0:8])
        tl_bbox = Float32MultiArray()
        if int(numboxes)>=1:
            tmp = -np.ones(7*int(numboxes)+1)

            for i in range(0,int(numboxes)):
                score = float(np.squeeze(scores_out)[i])
                bbox = [float(v) for v in np.squeeze(boxes_out)[i]]
                tmp[0] = numboxes
                if score > 0.3:
                    x_top_left = bbox[1] * cols
                    y_top_left = bbox[0] * rows
                    x_bottom_right = bbox[3] * cols
                    y_bottom_right = bbox[2] * rows
                    width = x_bottom_right - x_top_left
                    height = y_bottom_right - y_top_left 
                    tmp[5*i+1] = x_top_left
                    tmp[5*i+2] = y_top_left
                    tmp[5*i+3] = width
                    tmp[5*i+4] = height
                    tmp[5*i+5] = score
                    # tmp[6] = 1
                    # tmp[7] = .999


                    #########################################################333
                    # Adding color model into the tensorflow model
                    # # traffic_light = ["red", "green", "black"]
                    traffic_light =  ["black", "red", "yellow", "green"]

                    # mymodel = tf.keras.models.load_model("./saved_models/new_model/model_b_lr-4_ep150_ba32.h")
                    # frame = img[int(ymin):int(ymax), int(xmin):int(xmax)]
                    frame = image_np[int(y_top_left):int(y_bottom_right), int(x_top_left):int(x_bottom_right)]
                    # introdusing the color model here
                    # test_image = frame.resize((32, 32))
                    test_image = cv2.resize(frame, (32, 32))
                    # test_image = image
                    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
                    test_image = np.expand_dims(test_image, axis=0)
                    test_image /= 255
                    # print("My model is:" ,mymodel.summary())
                    # amodel=self.mymodel
                    with graph.as_default():
                        result = mymodel.predict(test_image)
                        # print("result is ", result)
                    # print("Prediction done")
                    # print(result)
                    #
                    tmp[6] = int(result.argmax())
                    tmp[7 * i + 7] =0.999
                    # tmp[7] = float(result[0, result.argmax()])

                    light_color = traffic_light[result.argmax()]
                    #
                    # light_color = light_color+"\n"
                    # selfight_co.append(light_color)
                    # with open('somefile4.txt', 'a') as the_file:
                    #     the_file.write(light_color)
                    # # '_______________________________________'

                    # cv2.imshow('images', frame)
                    # cv2.waitKey(1)
                    # cv2.destroyAllWindows()
                    # # cv2.waitKey(10)
                    # top = (int(x_top_left), int(y_top_left))
                    # bottom = (int(x_bottom_right), int(y_bottom_right))
                    # cv2.rectangle(image_np, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
                    # cv2.putText(image_np, light_color, (250, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),lineType=cv2.LINE_AA)
                    ################################################################
        else:
            tmp = [-1.0, 10.0, 10.0, 10.0, 10.0,10.0,10.0]

        # cv2.imshow("abcd", image_np)
        # cv2.waitKey(1)
        # output_path="./image5/"
        # ts = time.time()
        # st = datetime.datetime.fromtimestamp(ts).strftime('%f')
        # cut_frame = output_path
        # cut_frame = cut_frame + "image_" + str(st) + ".jpg"
        # print(cut_frame)
        # cv2.imwrite(cut_frame, image_np)

        tl_bbox.data = tmp
        thorth = time.time()
        time_cost = round((thorth-third)*1000)
        print("Inference Time",time_cost)
        print("bbox ",tl_bbox.data)
        self._pub.publish(tl_bbox)
# Spin once
def spin(self):
    rospy.spin()

def main():
    rospy.init_node('tf_object_detector', anonymous=True)
    tf_ob = ros_tensorflow_obj()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
if __name__ == '__main__':
    main()