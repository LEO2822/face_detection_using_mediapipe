
#- Importing all the libraries

import cv2 as cv
from mediapipe.python.solutions import face_mesh
import numpy as np
import mediapipe as mp

# Face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# to load the video
cap = cv.VideoCapture('./video.mp4')


while True:
      
      # Image

      ret , image = cap.read()
      if ret is not True:
            break
      height, width, _ = image.shape
      #print("(Height, Width) =", height, width)

      # NOTE :
      # opnecv process the image in bgr format. so we are converting this to rgb

      rgb_image  = cv.cvtColor(image, cv.COLOR_BGR2RGB)


      # Facial Landmarks
      # media pipe process the image in the rgb format
      result = face_mesh.process(rgb_image)

      # to get the focial_landmarks
      if result.multi_face_landmarks :
            for facial_landmarks in result.multi_face_landmarks:
                  # to show all the 468 landmarks
                  for i in range(0,468):
                        
                        # 0, shows the mouth (landmark)
                        # 1, shows the nose 
                        # and so on.. 
                        pt1 = facial_landmarks.landmark[i]
                  
                        # converting the points into coordinates and using int to make them interger
                        x = int(pt1.x * width)
                        y = int(pt1.y * height)
                        
                        # print("(x, y) = ", x, y)
                        
                        # to show/display the position of the facial_landmarks
                        cv.circle(image , (x, y) , 1 , (100,100,0) ,-1)
                        
                        # to check the index position of each landmark
                        #cv.putText(image , str(i) , (x,y) , 0 , 1 , (0,0,0))
                        
      # load the image
      cv.imshow("Image" , image)
      
      # waitkey(1) is to wait for 1ms and go to next frame
      if cv.waitKey(1) & 0xFF  == ord("q") :
            break
cv.destroyAllWindows()
cap.release()