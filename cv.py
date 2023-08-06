import cv2
import numpy as np

cap = cv2.VideoCapture()
while(1):
    # take each frame
    ret, frame = cap.read()
    if not ret:
        break

    # convert BDR to HSV
    hsv = cv2.cvtColor(frame, cv.CPLPR_BGR2HSV)

#=============== blue ===============
    #define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    #Threshold the HSV image to get only bule colors
    mark = cv2.inRange(hsv, lower_blue, upper_blue)
    blue = cv2.bitwise_and(frame,frame, mask= mask)

#=============== red ================
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    red = cv2.bitwise_and(frame,frame, mask= mask)

#=============== orange =============
    lower_orange = np.array([11,43,46])
    upper_orange = np.array([25,255,255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    orange  = cv2.bitwise_and(frame,frame, mask= mask)

#=============== yellow =============
    lower_yellow = np.array([26,43,46])
    upper_yellow = np.array([34,255,255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow  = cv2.bitwise_and(frame,frame, mask= mask)


#=============== green ==============
    lower_green = np.array([50,50,50])
    upper_green = np.array([70,255,255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green = cv2.bitwise_and(frame,frame, mask= mask)

#=============== gray ===============
    lower_gray = np.array([0,0,46])
    upper_gray = np.array([180,43,220])
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    gray = cv2.bitwise_and(frame,frame, mask= mask)

#=============== pink ================
    lower_gray = np.array([0,0,46])
    upper_gray = np.array([180,43,220])
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    gray = cv2.bitwise_and(frame,frame, mask= mask)

    #Bitewise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    k = cv2.waitKey(5)
    if k ==27:
        break




