
#- importing all the packages

import cv2
import mediapipe as mp
import numpy as np
import json

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def bounding_box(detection,h,w):
      
      '''to get the bounding box coordinates'''
      
      xmin = int(detection.location_data.relative_bounding_box.xmin*w)
      ymin = int(detection.location_data.relative_bounding_box.ymin*h)
      width = int(detection.location_data.relative_bounding_box.width*w)
      height = int(detection.location_data.relative_bounding_box.height*h)

      return xmin,ymin,width,height

# for json output
outer_dict ={}
frame_count=0

#- For webcam input:

cap = cv2.VideoCapture('./video.mp4')
with mp_face_detection.FaceDetection(
      model_selection=0, min_detection_confidence=0.5) as face_detection:
      while cap.isOpened():
            success, image = cap.read()
            h, w, c = image.shape
            if not success:
                  print("Ignoring empty camera frame.")
                  # If loading a video, use 'break' instead of 'continue'.
                  continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = face_detection.process(image)
            # print(results.detections)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            inner_dict={}
            if results.detections:
                  for idx,detection in enumerate(results.detections):
                        xmin,ymin,width,height = bounding_box(detection,h,w)
                        #new_face = image[ymin:ymin+height,xmin:xmin+width]
                        #resize_face = cv2.resize(new_face,(48,48))
                        gray_face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        img = cv2.merge([gray_face,gray_face,gray_face])
                        #img = img.reshape(-1,48,48,3)
                        # print(xmin,ymin,width,height)
                        cv2.rectangle(image, (xmin, ymin), (xmin+width, ymin+height), (0, 255, 0), 2)
                        # mp_drawing.draw_detection(image, detection)
                        # print(detection.location_data)
                        inner_dict["face_"+str(idx)]= {
                        "bounding_box":{"xmin":xmin,"ymin":ymin,"width":width,"height":height}
                        }
                        outer_dict["frame_"+str(frame_count)]= inner_dict
                        frame_count +=1
            cv2.imshow('MediaPipe Face Detection', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                  break
cap.release()

# creating json file
with open("video.json",'w') as f:
      json.dump(outer_dict,f,indent=4)