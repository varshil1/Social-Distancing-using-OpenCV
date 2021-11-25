# Importing libraries
import cv2 
import urllib
import requests,os
import numpy as np

from imutils.video import VideoStream, FPS
from scipy.spatial import distance as dist
import argparse, imutils, time
from mailer import Mailer

#Importing the functions
from detect_people import *

#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)
yolo_path = os.path.join(".\yolo")
# path for the YOLO directory
MODEL_PATH = ".\yolo"
# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3


""" CONFIGURATIONS """

# Counting the total number of people in the screen  
People_Counter = True

# Total maximum allowed violations of the social distancing  
Threshold = 15

# Enter the ip camera url (e.g., url = 'http://191.138.0.100:8040/video');
# Set url = 0 for webcam.
# url='http://192.168.29.213:8080/video'
# For any youtube video
# url='https://www.youtube.com/watch?v=GJNjaRJWVP8'
# url='http://10.20.129.49:8080/video'


# Want to send mail for threshold violations or not
ALERT = True

# Send the alerts to the given mail_id
MAIL = 'varsil.s@ahduni.edu.in'

# Define the max/min safe distance limits (in pixels) between 2 people.
MAX_DISTANCE = 80
MIN_DISTANCE = 50

""" Loading the data paths """

# load the COCO class labels our YOLO model was trained on
labels_Path = os.path.sep.join([MODEL_PATH, "coco.names"])
LABELS = open(labels_Path).read().strip().split("\n")

# load paths to the YOLO weights and model configuration
weights_Path =  os.path.sep.join([MODEL_PATH, "yolov3.weights"])

config_Path = os.path.sep.join([MODEL_PATH, "yolov3.cfg"])

# Loading the YOLO object detector
# readNetFromDarkNet : readNetFromDarknet() method to load the saved weights into the network. 
# This little command will give us the network architecture as specified in config loaded with the trained weights of yolov2.
net_yolo = cv2.dnn.readNetFromDarknet(config_Path, weights_Path)

# determine only the *output* layer names that we need from YOLO
ln = net_yolo.getLayerNames()
ln = [ln[i[0] - 1] for i in net_yolo.getUnconnectedOutLayers()]





"""Converting the video into frames"""

# Opening the video file
print("[INFO] Starting the video..")
video='videos/test.mp4'
vs = cv2.VideoCapture(video)





""" Writing the video into output file """

# Writing the output video

writer = None
# start the FPS counter
fps = FPS().start()

# looping over the frames from the video stream
while True:
  # read the next frame from the file
  (grabbed, frame) = vs.read()
  # if the frame was not grabbed, then we have reached the end of the stream
  # so break the frame
  if not grabbed:
    break

  # resize the frame and then detect people (and only people) in it
  frame = imutils.resize(frame, width=700)
  results = detect_people(frame, net_yolo, ln,
    personIdx=LABELS.index("person"))

  # initialize the set of indexes that violate the max/min social distance limits
  serious = set()
  abnormal = set()

  # ensure there are *at least* two people detections (required in
  # order to compute our pairwise distance maps)
  if len(results) >= 2:
    # extract all centroids from the results and compute the
    # Euclidean distances between all pairs of the centroids
    centroids = np.array([r[2] for r in results])
    D = dist.cdist(centroids, centroids, metric="euclidean")

    # loop over the upper triangular of the distance matrix
    for i in range(0, D.shape[0]):
      for j in range(i + 1, D.shape[1]):
        # check to see if the distance between any two
        # centroid pairs is less than the configured number of pixels
        if D[i, j] < MIN_DISTANCE:
          # update our violation set with the indexes of the centroid pairs
          serious.add(i)
          serious.add(j)
                # update our abnormal set if the centroid distance is below max distance limit
        if (D[i, j] < MAX_DISTANCE) and not serious:
          abnormal.add(i)
          abnormal.add(j)

  # loop over the results
  for (i, (prob, bbox, centroid)) in enumerate(results):
    # extract the bounding box and centroid coordinates, then
    # initialize the color of the annotation
    (startX, startY, endX, endY) = bbox
    (cX, cY) = centroid
    color = (0, 255, 0)

    # if the index pair exists within the violation/abnormal sets, then update the color
    if i in serious:
      color = (0, 0, 255)
    elif i in abnormal:
      color = (0, 255, 255) #orange = (0, 165, 255)

    # draw (1) a bounding box around the person and (2) the
    # centroid coordinates of the person,
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    cv2.circle(frame, (cX, cY), 5, color, 2)

  # draw some of the parameters
#   Safe_Distance = "Safe distance: >{} px".format(MAX_DISTANCE)
#   cv2.putText(frame, Safe_Distance, (470, frame.shape[0] - 25),
#     cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
#   Threshold_limit = "Threshold limit: {}".format(Threshold)
#   cv2.putText(frame, Threshold_limit, (470, frame.shape[0] - 50),
#     cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

    # draw the total number of social distancing violations on the output frame
  text = "Total violations: {}".format(len(serious))
  cv2.putText(frame, text, (10, frame.shape[0] - 55),
    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

  text1 = "Total warnings: {}".format(len(abnormal))
  cv2.putText(frame, text1, (10, frame.shape[0] - 25),
    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

# ------------------------------Alert function----------------------------------#
#   if len(serious) >= int(Threshold):
#     cv2.putText(frame, "-ALERT: Violations over limit-", (10, frame.shape[0] - 80),
#       cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 2)
#     if ALERT:
#       print("")
#       print('[INFO] Sending mail...')
      
#       mail = Mailer(email='systemchat000@gmail.com', password='ABCD@1234')
#       mail.send(receiver=MAIL, subject='Alert', message='Thredshold reached')
#       print('[INFO] Mail sent')
    # config.ALERT = False

    # update the FPS counter
  fps.update()

  # if an output video file path has been supplied and the video
  # writer has not been initialized, do so now
  if writer is None:
    # initialize our video writer
    # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    # writer = cv2.VideoWriter("output", fourcc, 25,
    #   (frame.shape[1], frame.shape[0]), True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter("output.mp4", fourcc, 25, (frame.shape[1], frame.shape[0]), True)
    # print("YESSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
    writer.write(frame)

  # if the video writer is not None, write the frame to the output video file
  if writer is not None:
    # print('yo')
    writer.write(frame)

# stop the timer and display FPS information
fps.stop()

# writer.write(frame)
writer.release()