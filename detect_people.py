# Importing libraries
import cv2 
import urllib
import requests,os
import numpy as np

from imutils.video import VideoStream, FPS
from scipy.spatial import distance as dist
import argparse, imutils, time

MIN_CONF = 0.3
NMS_THRESH = 0.3

# Counting the total number of people in the screen  
People_Counter = True

Threshold = 15
def detect_people(frame, net, ln, personIdx=0):

  """
  THis function takes the input as the frame from the video and find the persons in each frame , 
  draws the bounded boxes and detect the number of people in the frame as an output.

  Parameters:

  frame : frame obtained while reading the video
  net : Darknet which is used to train the YONO model
  ln :  *output* layer names that we need from YOLO
  personIdx : index of the label having the value 'person'
  """


  # Fetch the dimensions of the frame
  (Height, Width) = frame.shape[:2]
  # Initialize the list named result
  results = []

  # construct a blob from the input frame : creates 4-dimensional blob from image.

  # blob = cv2.dnn.blobFromImage(imageFrame, scalefactor, size, swapRB, crop)
    
        # *  @param image input image (with 1-, 3- or 4-channels).
        # *  @param size spatial size for output image
        # *  @param mean scalar with mean values which are subtracted from channels. Values are intended
        # *  to be in (mean-R, mean-G, mean-B) order if @p image has BGR ordering and @p swapRB is true.
        # *  @param scalefactor multiplier for @p image values.
        # *  @param swapRB flag which indicates that swap first and last channels
        # *  in 3-channel image is necessary.
        # *  @param crop flag which indicates whether image will be cropped after resize or not
        
  blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)

  # perform a forward
  # pass of the YOLO object detector,
  net.setInput(blob)

  # giving us our bounding boxes
  # and associated probabilities
  layerOutputs = net.forward(ln)

  # initialize our lists for detecting the bounding boxes, centroids, and
  # confidences, respectively
  boxes = []
  centroids = []
  confidences = []

  # Iterating loop over each of the layer outputs
  for output in layerOutputs:
    # Iterating loop over each of the detections
    for detection in output:
      # extract the class ID and confidence (i.e., probability)
      # of the current object detection
      # Extract Top 5 scores of current object detection
      scores = detection[5:]
      # Getting the ID of the maximum confident detected object
      class_ID = np.argmax(scores)
      # Probability of the object
      confidence = scores[class_ID]

      # Dectecting the person in the fram
      # Ensure that the object is person and the confidence is more than the min required
      if class_ID == personIdx and confidence > MIN_CONF:
        # scale the bounding box coordinates back relative to
        # the size of the image, keeping in mind that YOLO
        # actually returns the center (x, y)-coordinates of
        # the bounding box followed by the boxes' width and
        # height
        box = detection[0:4] * np.array([Width, Height, Width, Height])
        (centerX, centerY, width, height) = box.astype("int")

        # Getting top left corner of the bounding boxes
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))

        # update our list of bounding box coordinates,
        # centroids, and confidences
        boxes.append([x, y, int(width), int(height)])
        centroids.append((centerX, centerY))
        confidences.append(float(confidence))


  #NMSBoxes : Performs non maximum suppression given boxes and corresponding scores.

  #  * @param bboxes a set of bounding boxes to apply NMS.
  #  * @param scores a set of corresponding confidences.
  #  * @param score_threshold a threshold used to filter boxes by score.
  #  * @param nms_threshold a threshold used in non maximum suppression.
  #  * @param indices the kept indices of bboxes after NMS.

  # apply non-maxima suppression to suppress weak, overlapping
  # bounding boxes
  idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

  #idxs gives total number of boxes 
  #print('Total people count:', len(idxs))
  # compute the total people counter
  if People_Counter:
    human_count = "Human count: {}".format(len(idxs))
    cv2.putText(frame, human_count, (470, frame.shape[0] - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 2)

  # ensure at least one detection exists
  if len(idxs) > 0:
    # Iterating the loop over the indexes we are keeping
    for i in idxs.flatten():
      # extract the bounding box coordinates
      (x, y) = (boxes[i][0], boxes[i][1])
      (w, h) = (boxes[i][2], boxes[i][3])

      # update our results list to consist of the person
      # prediction probability, bounding box coordinates,
      # and the centroid
      r = (confidences[i], (x, y, x + w, y + h), centroids[i])
      results.append(r)

  # return the list of results
  return results
