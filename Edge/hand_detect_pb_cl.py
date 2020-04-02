# organize imports
import cv2
import numpy as np
import copy
import math
import os 
import tensorflow as tf
import sys
import time

# global variables
bg = None

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 70
learningRate = 0

frame_rate = 5
prev = 0

# ASL signs
gesture_names = {0: 'A',
                 1: 'B',
                 2: 'C',
                 3: 'D',
                 4: 'del',
                 5: 'E',
                 6: 'F',
                 7: 'G', 
                 8: 'H',
                 9: 'I',
                 10: 'J',
                 11: 'K',
                 12: 'L',
                 13: 'M',
                 14: 'N',
                 15: 'nothing', 
                 16: 'O',
                 17: 'P',
                 18: 'Q',
                 19: 'R',
                 20: 'S',
                 21: 'space', 
                 22: 'T', 
                 23: 'U',
                 24: 'V',
                 25: 'W',
                 26: 'X',
                 27: 'Y',
                 28: 'Z'}


# load the pb file 
PATH_TO_FROZEN_GRAPH = './saved_model'
#PATH_TO_LABEL_MAP = './labels.pbtxt'

# create word and text:
pred_list = ['','']
letters = ''
word = ''
text = []



# number of classes 
NUM_CLASSES = 29

detection_graph = tf.saved_model.load(PATH_TO_FROZEN_GRAPH)
infer = detection_graph.signatures["serving_default"]
print(infer.inputs)
print(infer.outputs)


#--------------------------------------------------
# Help functions
#--------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(1)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 234, 574

    # initialize num of frames
    num_frames = 0

    # keep looping, until interrupted
    # REF: https://stackoverflow.com/questions/52068277/change-frame-rate-in-opencv-3-4-2
    while(True):
        # get the current frame and set fps
        time_elapsed = time.time() - prev
        (grabbed, frame) = camera.read()

        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)     

        if time_elapsed > 1./frame_rate:
            prev = time.time()

               
        # img = removeBG(frame)
        # img = img[0:int(cap_region_y_end * frame.shape[0]),int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI

        # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)

        # clone the frame
            clone = frame.copy()

        # get the height and width of the frame
            (height, width) = frame.shape[:2]

        # get the ROI
            roi = frame[top:bottom, right:left]
            roi = removeBG(roi)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(roi, (224, 224))
            # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            cv2.imshow("roi", roi)
	
        # draw the segmented hand
            cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # sign prediction
            extra_dim = tf.expand_dims(roi, 0)
            x = tf.constant(extra_dim, dtype=tf.uint8)
            casted = tf.dtypes.cast(x, tf.float32)  # [1, 2], dtype=tf.int32 
            labeling = infer(tf.constant(casted))
            print("Result after saving and loading:\n", labeling)
            # print(labeling.get('outputs'))
            max_pred = np.argmax(labeling.get('outputs'))
            pred_sign = gesture_names.get(max_pred)
            
            if pred_sign == 'nothing':
                letters = letters + str(pred_list[-1])
                pred_list = ['']
                print(letters)
                pass
            elif pred_sign != 'nothing' and pred_sign != 'space':
                pred_list.append(pred_sign)
                print(pred_list)
            elif pred_sign == 'space':
                pred_list.append(' ')
                
        # display the frame with segmented hand
        # REF: https://stackoverflow.com/questions/50854235/how-to-draw-chinese-text-on-the-image-using-cv2-puttextcorrectly-pythonopen
            cv2.imshow("Video Feed", clone)
            
        # print text on screen
            image = np.zeros((200,800,3),np.uint8)
            b,g,r,a = 0,255,0,0
            cv2.putText(image,  letters, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b,g,r), 1, cv2.LINE_AA)
            cv2.imshow("Pred_sign", image)

        # observe the keypress by the user
            keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
            if keypress == ord("q"):
                break

# free up memory
camera.release()
cv2.destroyAllWindows()

