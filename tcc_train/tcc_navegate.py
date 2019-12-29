#! /home/david/tcc_project/bin/python
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import os

from time import time

from std_msgs.msg import String
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist

# import the necessary packages
from keras.models import load_model
import argparse
import pickle

import tensorflow as tf


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#     help="path to input image we are going to classify")
ap.add_argument("-m", "--model", required=True,
    help="path to trained Keras model")
ap.add_argument("-l", "--label-bin", required=True,
    help="path to label binarizer")
ap.add_argument("-w", "--width", type=int, default=28,
    help="target spatial dimension width")
ap.add_argument("-e", "--height", type=int, default=28,
    help="target spatial dimension height")
ap.add_argument("-f", "--flatten", type=int, default=-1,
    help="whether or not we should flatten the image")
args = vars(ap.parse_args())

# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model(args["model"])
model._make_predict_function()
graph = tf.get_default_graph()
lb = pickle.loads(open(args["label_bin"], "rb").read())

time_initial = 0
time_final = 0
qtd_moves = 0
c_move = 0
l_move = 0
r_move = 0

# Topics for controle drone
# takeoff
takeoff_pub = rospy.Publisher("ardrone/takeoff", Empty, queue_size=10 )
# land
land_pub = rospy.Publisher("ardrone/land", Empty, queue_size=10 )
# Control drone
control_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10 )

# fly forward
# control_pub.publish('{linear:  {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0,y: 0.0,z: 0.0}}')
# fly backward
# rostopic pub -r 10 /cmd_vel geometry_msgs/Twist  '{linear:  {x: -1.0, y: 0.0, z: 0.0}, angular: {x: 0.0,y: 0.0,z: 0.0}}'
# fly to left
# rostopic pub -r 10 /cmd_vel geometry_msgs/Twist  '{linear:  {x: 0.0, y: 1.0, z: 0.0}, angular: {x: 0.0,y: 0.0,z: 0.0}}'
# fly to right
# rostopic pub -r 10 /cmd_vel geometry_msgs/Twist  '{linear:  {x: 0.0, y: -1.0, z: 0.0}, angular: {x: 0.0,y: 0.0,z: 0.0}}'
# fly up
# rostopic pub -r 10 /cmd_vel geometry_msgs/Twist  '{linear:  {x: 0.0, y: 0.0, z: 1.0}, angular: {x: 0.0,y: 0.0,z: 0.0}}'
# fly down
# rostopic pub -r 10 /cmd_vel geometry_msgs/Twist  '{linear:  {x: 0.0, y: 0.0, z: -1.0}, angular: {x: 0.0,y: 0.0,z: 0.0}}' 

# Instantiate CvBridge
bridge = CvBridge()
count2 = 0
count = 847
def image_callback(msg):
    global count
    global count2
    if(count2<15):
        count2+=1
        return
    else:
        count2=0
    print("Received an image!" + str(count))
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        count+=1
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg 
        print('Called....')        
        # cv2.imwrite( os.path.join(os.path.expanduser('~'),'Desktop','test_'+str(count)+'.jpeg'), cv2_img)
        check_image(cv2_img)

def check_image(img):
    global model
    global lb
    global graph
    # load the input image and resize it to the target spatial dimensions
    image = img
    output = image.copy()
    image = cv2.resize(image, (args["width"], args["height"]))

    # scale the pixel values to [0, 1]
    image = image.astype("float") / 255.0

    # check to see if we should flatten the image and add a batch
    # dimension
    if args["flatten"] > 0:
        image = image.flatten()
        image = image.reshape((1, image.shape[0]))

    # otherwise, we must be working with a CNN -- don't flatten the
    # image, simply add the batch dimension
    else:
        image = image.reshape((1, image.shape[0], image.shape[1],
            image.shape[2]))

    with graph.as_default():
        # make a prediction on the image
        preds = model.predict(image)

        # find the class label index with the largest corresponding
        # probability
        i = preds.argmax(axis=1)[0]
        label = lb.classes_[i]

        # draw the class label + probability on the output image
        text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
        print(text)
        # cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        #     (0, 0, 255), 2)

        # show the output image
        # cv2.imshow("Image", output)
        # rospy.sleep(1)
        navegate(label)
        
        print('next')
        
        # cv2.waitKey(0)

def navegate(dir):
    global control_pub
    global takeoff_pub
    global time_final
    global qtd_moves
    global c_move
    global l_move
    global r_move
    time_final = time()
    qtd_moves+=1
    takeoff_pub.publish(Empty())
    vel_msg = Twist()
    vel_msg.angular.x = 0.0
    vel_msg.angular.y = 0.0
    vel_msg.angular.z = 0.0

    if(dir == 'center'):
        vel_msg.linear.x = 0.5
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        c_move +=1
        print('move to center')
    elif(dir == 'left'):
        vel_msg.linear.x = 0.0
        vel_msg.linear.y = -0.3
        vel_msg.linear.z = 0.0
        r_move +=1
        print('move to left')
    elif(dir == 'right'):
        vel_msg.linear.x = 0.0
        vel_msg.linear.y = 0.3
        vel_msg.linear.z = 0.0
        l_move +=1
        print('move to right')
    else:
        print('stop')
    print("QTD_MOVE: {:d}, C: {:d}, R: {:d}, L: {:d}".format(qtd_moves, c_move, r_move, l_move))

    control_pub.publish(vel_msg)

    rospy.sleep(0.5)
    vel_msg.linear.x = 0.0
    vel_msg.linear.y = 0.0
    vel_msg.linear.z = 0.0
    control_pub.publish(vel_msg)

def main():
    global time_initial
    global takeoff_pub
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/ardrone/front/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    
    
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()

# USAGE
# Stop rostopic pub -1 /ardrone/land std_msgs/Empty
# Start python tcc_navegate.py --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --width 32 --height 32 --flatten 1
# Start cnn python tcc_navegate.py --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle --width 64 --height 64
# Camera rosrun image_view image_view image:=/ardrone/front/image_raw 
# World roslaunch cvg_sim_gazebo tcc_world.launch

# python predict.py --image images/dog.jpg --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --width 32 --height 32 --flatten 1
# python predict.py --image images/dog.jpg --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle --width 64 --height 64