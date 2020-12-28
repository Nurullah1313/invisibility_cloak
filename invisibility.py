import cv2
import numpy as np

cam = cv2.VideoCapture(0)
cv2.namedWindow('invisible',cv2.WINDOW_NORMAL)

def morpholofy(mask):
    kernel1  = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    kernel2  = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))
    erode = cv2.erode(mask,kernel1,iterations = 1)
    dilation = cv2.dilate(erode, kernel2, iterations = 1)
    opening = cv2.morphologyEx(dilation,cv2.MORPH_OPEN,kernel1,iterations = 1)
    closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel2,iterations = 2)

    return closing

def hsv_al(img):

    l_v = np.array([131, 60,  57])
    h_v = np.array([155, 255, 255])

    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv,l_v,h_v)
    res = cv2.bitwise_and(img,img,mask = mask)

    return res, mask

_, background = cam.read()

def invisible(frame,mask):
    global background
    mask_not = cv2.bitwise_not(mask)

    back = cv2.bitwise_or(background,background,mask= mask)  
    extraction = cv2.bitwise_and(frame,frame, mask = mask_not)
    dst = cv2.addWeighted(extraction,1,back,1,0)
    
    return dst


while True:
    _, frame = cam.read()

    res,mask = hsv_al(frame)
    dilation = morpholofy(mask)
    dst = invisible(frame,dilation)

    cv2.namedWindow('invisible',cv2.WINDOW_NORMAL)
    cv2.imshow('invisible',dst)


    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

