"""
Created on 5-23-2025
@author: Jenish Patel

"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import os
import argparse

class FaceDetect(Node):
    def __init__(self):
        super().__init__('face_detect_node')
        self.bridge = CvBridge()
        self.cascade_path = "/home/jpat/ros2face/src/facedetectapril/script/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        if self.face_cascade.empty():
            print("Error loading cascade file")
            sys.exit(1)

    def receive_image(self):    
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.frame_callback,
            10
        )
        self.get_logger().info("Face detection node has been started")
        rclpy.spin(self)
        cv2.destroyAllWindows()
    
    def frame_callback(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        if frame is None:
            self.get_logger().error("Received an empty frame")
            return

        #Turns the frame into grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        #Creates a bounding box around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imshow("Face Detection", frame)
        cv2.waitKey(1)
    
def main(args=None):
    rclpy.init(args=args)
    node = FaceDetect()
    node.receive_image()

if __name__ == '__main__':
    main()

