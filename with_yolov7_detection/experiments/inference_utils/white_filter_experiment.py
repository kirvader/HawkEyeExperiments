import cv2
import numpy as np

video_input_source = "/home/kir/hawk-eye/HawkEyeExperiments/video_trimmer/videos/3.mp4"

cap = cv2.VideoCapture(video_input_source)

while True:
    status, frame = cap.read()
    if not status:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([0, 0, 210], dtype=np.uint8)
    upper_white = np.array([179, 51, 255], dtype=np.uint8)
    # lower_white = np.array([0,0,0], dtype=np.uint8)
    # upper_white = np.array([0,0,255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break