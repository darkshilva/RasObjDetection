#!/usr/bin/env python
# coding: utf-8

# In[170]:


######## Picamera Object Detection Using Tensorflow #########
#
# Author: Quang Khoi Tran
# Date: 11/03/19
# Description: 
# This program uses TensorFlow to perform object detection.
# It loads the classifier uses it to perform object detection on a Picamera feed.
# It draws boxes and scores around the objects of interest in each frame from
# the Picamera. 
## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py


# In[171]:


# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
import time
import psutil as sys_info 


# In[172]:


# Set up images resolution
IM_WIDTH = 1280
IM_HEIGHT = 720
#IM_WIDTH = 640    
#IM_HEIGHT = 480   


# In[173]:


# Select camera type (trying feature: if user enters --webcam when calling this script,
# a webcam will be used)
camera_type = 'picamera'
''' #testing feature to run program from terminal with --webcam
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'webcam'
 '''
#camera_type='webcam'
# This is needed since the working directory is the object_detection folder.
sys.path.append('..')


# In[174]:


# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


# In[175]:


# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90


# In[176]:


## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `aircraft`.
# This code use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# In[177]:


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


# In[178]:



# Initialize frame rate calculation
frame_rate = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX


# In[179]:





# In[180]:


#Initialize time logging
time_local = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", time_local)

#Initialize cpu and memory using logging
sys_info_cpu=sys_info.cpu_percent()
sys_info_ram=sys_info.virtual_memory()[2] #2 = percent


# In[181]:


# piCamera
if camera_type == 'picamera':
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    #Initialize output video
    # VideoWriter is the responsible of creating a copy of the video
    # used for the detections but with the detections overlays. Keep in
    # mind the frame size has to be the same as original video.
    out = cv2.VideoWriter('object_detected_result_pi.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (IM_WIDTH, IM_HEIGHT))
    #VideoWriter('name.avi,codec,FPS,(width,height)')
    
    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
        cv2.putText(frame,time_string,(250,50),font,1,(255,255,0),2,cv2.LINE_AA)
        cv2.putText(frame,"CPU: {0:.2f}".format(sys_info_cpu),(750,50),font,1,(255,255,0),2,cv2.LINE_AA)
        cv2.putText(frame,"RAM: {0:.2f}".format(sys_info_ram),(950,50),font,1,(255,255,0),2,cv2.LINE_AA)
     

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)
        #write the result video
        # https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/
        #color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #output_rgb = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate = 1/time1 #update Frame rate
        time_local = time.localtime() # update local time
        time_string = time.strftime("%m/%d/%Y, %H:%M:%S", time_local)
        sys_info_cpu=sys_info.cpu_percent() #update cpu and ram using
        sys_info_ram=sys_info.virtual_memory()[2]

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    out.realease()
    camera.close()


# In[182]:





# In[ ]:





# In[ ]:




