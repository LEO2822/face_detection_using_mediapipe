
# Importing modules
import sys
import cv2
import mediapipe as mp
import json

# Initialize mediapipe face detector and drawer
mp_face_detection = mp.solutions.face_detection
mp_face_drawing   = mp.solutions.drawing_utils

image_path = []

def bounding_box(detection,h,w):
	xmin = int(detection.location_data.relative_bounding_box.xmin*w)
	ymin = int(detection.location_data.relative_bounding_box.ymin*h)
	width = int(detection.location_data.relative_bounding_box.width*w)
	height = int(detection.location_data.relative_bounding_box.height*h)

	return xmin,ymin,width,height

outer_dict = {}
frame_count = 0

# Detect faces on an image_path


	# Using mediapipe face detector
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
	
	# Load the images
	img = cv2.imread(image_path)
	h, w, c = img.shape
	# Convert the BGR image_path to RGB
	img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# Get faces on the an image_path (frame)
	faces = face_detection.process(img_grey)

	# Convert image_path back to its colors for displaying
	img_final = cv2.cvtColor(img_grey, cv2.COLOR_RGB2BGR)

	inner_dict = {}

	# Draw the face detection annotations on the image_path
	if faces.detections:
		for idx,detection in enumerate(faces.detections):
			xmin,ymin,width,height = bounding_box(detection,h,w)
			new_face = image_path[ymin:ymin+height,xmin:xmin+width]
			resize_face = cv2.resize(new_face,(48,48))
			gray_face = cv2.cvtColor(resize_face, cv2.COLOR_BGR2GRAY)
			img = cv2.merge([gray_face,gray_face,gray_face])
			img = img.reshape(-1,48,48,3)
			# print(xmin,ymin,width,height)
			cv2.rectangle(image_path, (xmin, ymin), (xmin+width, ymin+height), (0, 255, 0), 2)
			# mp_drawing.draw_detection(image_path, detection)
			# print(detection.location_data)
			inner_dict["face_"]= {
			"bounding_box":{"xmin":xmin,"ymin":ymin,"width":width,"height":height}
			}
			outer_dict["frame_"]= inner_dict
			
		cv2.imshow('MediaPipe Face Detection', image_path)
		cv2.waitKey(0)


cv2.destroyAllWindows()
with open("recog.json",'w') as f:
      json.dump(outer_dict,f,indent=4)

#detectFromImg('./1.jpg')