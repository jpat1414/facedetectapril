"""
Created on 5-23-2025
@author: Jenish Patel

"""

import cv2
import numpy as np
import sys
import os
import argparse


record_dir = 'recording'
if not os.path.exists(record_dir):
    os.makedirs(record_dir)

'''
# initializing the object tracker
object_tracker = cv2.legacy.TrackerCSRT_create()
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    raise Exception("Could not open video device")

ret, frame = webcam.read()
if not ret:
    raise Exception("Could not find video device")

print("object tracker initialized")


framewidth = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
frameheight = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
framecount = int(webcam.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"m", "p", "4", "v")
recording = 'recording/' + 'video.mp4'
output = cv2.VideoWriter(recording, fourcc, 20.0, (framewidth, frameheight))
if not output.isOpened():
    raise Exception("Could not open video file")
'''
