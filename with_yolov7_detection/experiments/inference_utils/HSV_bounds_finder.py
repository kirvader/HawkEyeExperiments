import cv2
import sys
import numpy as np


def nothing(x):
    pass


# Load in image
# image = cv2.imread('img_close_range.png') #  (hMin = 0 , sMin = 0, vMin = 189), (hMax = 108 , sMax = 41, vMax = 255)
# image = cv2.imread('img_far_range.png') #  (hMin = 0 , sMin = 0, vMin = 187), (hMax = 41 , sMax = 53, vMax = 255)
image = cv2.imread('img_mid_range.png')  # (hMin = 0 , sMin = 0, vMin = 161), (hMax = 179 , sMax = 43, vMax = 255)
# image = cv2.imread('img_shadowed_place.png') #  (hMin = 98 , sMin = 45, vMin = 77), (hMax = 179 , sMax = 117, vMax = 193)
# image = cv2.imread('img_with_shadow.png') #   (hMin = 0 , sMin = 0, vMin = 50), (hMax = 94 , sMax = 46, vMax = 255)
# for contours (hMin = 1 , sMin = 6, vMin = 152), (hMax = 179 , sMax = 39, vMax = 238)

# Create a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)  # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

output = image
wait_time = 33

while (1):

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')

    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    blurred = cv2.medianBlur(mask, 25)

    # Print if there is a change in HSV value
    if ((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax)):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (
        hMin, sMin, vMin, hMax, sMax, vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    ret, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Display output image
    cv2.imshow('image', thresh1)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
