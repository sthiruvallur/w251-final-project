# REF: https://github.com/athena15/project_kojak/blob/master/real_time_gesture_detection.py

#! /usr/bin/env python3

# load packages
import copy
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from gtts import gTTS 
import os 

# General Settings for text
text = ''

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


# load the pre-trained model (.pb files) 
PATH_TO_FROZEN_GRAPH = './SavedModel'
detection_graph = tf.saved_model.load(PATH_TO_FROZEN_GRAPH)
infer = detection_graph.signatures["serving_default"]
# print(infer.inputs)
# print(infer.outputs)

# set the directory for mp3 saving
DIR = r'/tmp/hands'
os.chdir(DIR)

# parameters
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variableslt
isBgCaptured = 0  # bool, whether the background captured

#--------------------------------------------------
# Help functions
#--------------------------------------------------

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

#-----------------
# MAIN FUNCTION
#-----------------

# Camera
camera = cv2.VideoCapture(0)
camera.set(10, 200)

while camera.isOpened():
    ret, frame = camera.read()

    # apply bilateral filter with d = 5 
    # sigmaColor = 50
    # sigmaSpace = 100
    frame = cv2.bilateralFilter(frame, 5, 50, 100) 
    # flip the frame horizontally
    frame = cv2.flip(frame, 1)  
    # mark ROI
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    # show original frame
    cv2.imshow('Frame image', frame)

    # Run once background is captured
    if isBgCaptured == 1:
        img = frame 
        
        # crop ROI
        img = img[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        # cv2.imshow('mask', img)

        # convert the image into RGB image
        cv2.imshow('ROI image', img)
        # resize ROI for prediction 
        roi = cv2.resize(img, (224, 224))
        
        
    # Keyboard OP
    k = cv2.waitKey(1)

    # press ESC to exit all windows at any time
    if k == 27:  
        break

    # press 'b' to capture the background
    elif k == ord('b'):  
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        time.sleep(2)
        isBgCaptured = 1
        print('Background captured')

    # press 'c' to clean the text
    elif k == ord('c'):
        letters = ''

    # press 'r' to reset the background
    elif k == ord('r'):  
        time.sleep(1)
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print('Reset background')

    # press 'space bar' to predict sign language
    elif k == 32:
        extra_dim = tf.expand_dims(roi, 0)
        x = tf.constant(extra_dim, dtype=tf.uint8)
        casted = tf.dtypes.cast(x, tf.float32) 
        
        # perform letter prediction
        labeling = infer(tf.constant(casted))
        print("Result after saving and loading:\n", labeling)

        # extract the letter with the largest probability
        max_pred = np.argmax(labeling.get('outputs'))
        pred_acc = round(np.amax(labeling.get('outputs')), 3)
        pred_sign = gesture_names.get(max_pred)
        print("--------------------------------")
        print("Predict letter: %s" %str(pred_sign))
        print("Predict probability: %s" %str(pred_acc))
        print("--------------------------------")

        # generate sentance from predicted letter

        if pred_sign == 'nothing': # delete last letter
            text = text[:-1]

        elif pred_sign != 'nothing' and pred_sign != 'space': # add letter to sentance
            text = text + str(pred_sign)

        elif pred_sign == 'space': # add space to sentance
            text = text + ' '
                
        # display the frame with segmented hand
        # REF: https://stackoverflow.com/questions/50854235/how-to-draw-chinese-text-on-the-image-using-cv2-puttextcorrectly-pythonopen
        image = np.zeros((200,600,3),np.uint8)
        b,g,r,a = 0,255,0,0
        cv2.putText(image,  text, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b,g,r), 1, cv2.LINE_AA)
        cv2.imshow("Pred_sign", image)

    # Press a to record audio.mp3, since TX2 doesn't have audio output
    # REF:https://www.geeksforgeeks.org/convert-text-speech-python/
    elif k == ord("a"):
        audio = gTTS(text=text,lang="en", slow=False)
        audio.save(text+".mp3")
        
# free up memory
camera.release()
cv2.destroyAllWindows()