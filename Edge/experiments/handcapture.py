# REF: https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d#695b
# REF: https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
# REF: https://stackoverflow.com/questions/58461211/saved-model-from-automl-vision-edge-not-loading-properly
# REF: https://www.tensorflow.org/guide/saved_model
# REF: https://github.com/lzane/Fingers-Detection-using-OpenCV-and-Python/tree/py2_opencv2
# REF: https://github.com/athena15/project_kojak/blob/master/real_time_gesture_detection.py

import cv2
import numpy as np
import copy
import math
import os 
import tensorflow as tf
import sys
#from utils import label_map_util
#from utils import visualization_utils as vis_util

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# load the pb file 
PATH_TO_FROZEN_GRAPH = './SavedModel'
#PATH_TO_LABEL_MAP = './labels.pbtxt'

# number of classes 
NUM_CLASSES = 29

detection_graph = tf.saved_model.load(PATH_TO_FROZEN_GRAPH)
infer = detection_graph.signatures["serving_default"]
print(infer.inputs)
print(infer.outputs)

#probability_model = tf.keras.Sequential([detection_graph, tf.keras.layers.Softmax()])
#label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
#categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#category_index = label_map_util.create_category_index(categories)

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))

def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0


# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
#cv2.namedWindow('trackbar')
#cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)

directory = r'/tmp/hands'
os.chdir(directory)

while(True):
    ret, frame = camera.read()
    #threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)
    #  Main operation
    bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)        
    img = removeBG(frame)
    img = img[0:int(cap_region_y_end * frame.shape[0]),int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
           
    # convert the image into binary image
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    resized = cv2.resize(img,(224,224))
    print(resized.shape)
    #flag, bts = cv2.imencode('.jpg', blur)
    #inp = [bts[:,0].tobytes()]
    #cv2.imshow('blur', blur)
    #ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fname = "frame3.jpg"
    cv2.imwrite(fname, resized)
    #print("captured hands!")
    #out = infer(key=tf.constant('something_unique'), image_bytes=tf.constant(inp))
    #REF: https://www.tensorflow.org/api_docs/python/tf/expand_dims
    extra_dim = tf.expand_dims(resized, 0)
    x = tf.constant(extra_dim, dtype=tf.uint8)
    casted = tf.dtypes.cast(x, tf.float32)  # [1, 2], dtype=tf.int32 
    labeling = infer(tf.constant(casted))
    print("Result after saving and loading:\n", labeling)
                

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
 
