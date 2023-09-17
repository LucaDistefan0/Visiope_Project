import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np
import picamera
import io
import tempfile
import time
import os
import cv2
import imutils
from imutils.object_detection import non_max_suppression
from tensorflow.keras.preprocessing.image import img_to_array
import sys
import importlib.util

IMAGE_PATH ='/home/pi/Visiope/DATASET4'
INPUT_SIZE = (350, 350)
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (224, 224)
BATCH_SIZE = 64
min_confidence = 0.5
visualize_debug = -1

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #rescale=1./255,
    #shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    #keep_aspect_ratio=True,
    validation_split=0.33
    )
IMG_SIZE=(224, 224)

train_generator = train_datagen.flow_from_directory(
    IMAGE_PATH,
    target_size=IMG_SIZE,
    batch_size=32,
    subset='training')
class_names = train_generator.class_indices
valid_generator = train_datagen.flow_from_directory(
    IMAGE_PATH,
    target_size=IMG_SIZE,
    batch_size=16,
    subset='validation')

#Model acquisition
model = tflite.Interpreter('/home/pi/Visiope/model_tflite/model_tflite')
input_details=model.get_input_details()
input_index=input_details[0]['index']
model.resize_tensor_input(input_index, [64,224,224,3])
model.allocate_tensors()
output_details=model.get_output_details()




"""# SLIDING WINDOWS + IMAGE PYRAMIDS FUNCTIONS

"""

def sliding_window(image, step, ws):
    # slide a window across the image
    for y in range(0, image.shape[0]-ws[1], step):
        for x in range(0, image.shape[1]-ws[0], step):
            # yield the current window
            yield (x, y, image[y:y+ws[1], x:x+ws[0]])

def image_pyramid(image, scale=1.5, minSize=(224, 224)):
    # yield the original image
    yield image

    # keep looping over the image pyramid
    while True:
        # compute the dimensions of the next image in the pyramid
        w = int(image.shape[1]/scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied mininum size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
           break

        # yield the next image in the pyramid
        yield image

def classify_batch(model, batchROIs, batchLocs, labels, minProb=0.2, top=10, dims=(224,224)):
    # pass our batch ROIs through our network and decode the predictions
    batchROIs = np.reshape(batchROIs, (64,224,224,3))
    model.set_tensor(input_details[0]['index'],batchROIs)
    model.invoke()
    preds = model.get_tensor(output_details[0]['index'])
    
    # dictionary building
    P = []
    chiavi = list(class_names.keys())
    for i in range(BATCH_SIZE):
      P_tmp = {}
      for j in range(len(class_names)):
        P_tmp[chiavi[j]] = preds[i][j]
      P.append(P_tmp)

    # loop over the decoded predictions
    i=0
    count = 0
    for i in P:
      for (k,v) in i.items() :
        # filter out weak detections by ensuring the predicted probability is greater than the minimum probability
        if v > minProb:
          # grab the coordinates of the sliding window for prediction and construct the bounding box
          (pX, pY) = batchLocs[count]
          box = (pX, pY, pX+dims[0], pY+dims[1])
          # grab the list of predictions for the label and add the bounding box + probability to the list
          L = labels.get(k, [])
          L.append((box, v))
          labels[k] = L
      count = count + 1
    
    return labels


labels = {}

camera=picamera.PiCamera()

while True:
    
    camera.capture("/home/pi/Visiope/Pict/img1.jpg")
    frame='/home/pi/Visiope/Pict/img1.jpg'
    
    orig = cv2.imread(frame)
    orig = cv2.resize(orig,(224,224))
    
    cv2.imshow('Webcam',orig)

    # load the input image from disk and grab its dimensions
    (h,w) = orig.shape[:2]

    # resize the input image to be a square
    resized = cv2.resize(orig, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)

    # initialize the batch ROIs and (x,y) coordinates
    batchROIs = None
    batchLocs = []

    # start the timer
    print("[INFO] detecting objects...")
    start = time.time()

    # loop over the image pyramid
    for image in image_pyramid(resized, scale=PYR_SCALE, minSize=ROI_SIZE):
        # loop over the sliding window locations
        for (x,y,roi) in sliding_window(resized, WIN_STEP, ROI_SIZE):
            # take the ROI and pre_process it so we can later classify the region with Keras
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            roi = roi / 255.0

            if batchROIs is None:
                batchROIs = roi
            else:
                batchROIs = np.vstack([batchROIs, roi])
            batchLocs.append((x,y))

            # check to see if our batch is full
            if len(batchROIs) == BATCH_SIZE:
                # classify the batch, then reset the batch ROIs
                # and (x,y)-coordinates
                labels = classify_batch(model, batchROIs, batchLocs, labels)
                batchROIs = None
                batchLocs = []

    # check to see if there are any remaining ROIs that still need to be classifier
    if batchROIs is not None:
        labels = classify_batch(model, batchROIs, batchLocs, labels)

    # show how long the detection process took
    end = time.time()
    print("[INFO] detections took {:.4f} seconds".format(end-start))


    for k in labels.keys():
        # clone the input image so we can draw on it
        clone = resized.copy()
        # loop over all bounding boxes for the label and draw them on the image
        for(box, prob) in labels[k]:
            (xA, yA, xB, yB) = box
            cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # show the image *without* apply non-maxima suppression
        cv2.imshow('without nmp', clone)
        cv2.imwrite('/home/pi/Visiope/clone.png',clone)
        clone = resized.copy()

        # grab the bounding boxes and associated probabilities for each detection,
        # then apply non-maxima suppression to suppress weaker, overlapping detections
        boxes = np.array([p[0] for p in labels[k]])
        proba = np.array([p[1] for p in labels[k]])
        boxes = non_max_suppression(boxes, proba)

        # loop over the bounding boxes again, this time only drawing the ones that were not supressed
        for (xA, yA, xB, yB) in boxes:
            cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 0, 255), 2)

        # show the output image
        print("[INFO] {}: {}".format(k, len(boxes)))
        cv2.imshow('with nms', clone)
        cv2.waitKey(0)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()

