# Importing modules
import sys
import cv2
import mediapipe as mp

# Initialize mediapipe face detector and drawer
mp_face_detection = mp.solutions.face_detection
mp_face_drawing   = mp.solutions.drawing_utils

# Detect faces on an image
def detectFromImg(image_path):
	"""Function to detect faces on an image"""

	# Using mediapipe face detector
	with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
		
		# Load the images
		img = cv2.imread(image_path)
    
		# Convert the BGR image to RGB
		img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# Get faces on the an image (frame)
		faces = face_detection.process(img_grey)

		# Convert image back to its colors for displaying
		img_final = cv2.cvtColor(img_grey, cv2.COLOR_RGB2BGR)

		# Draw the face detection annotations on the image
		if faces.detections:
			for detection in faces.detections:
				mp_face_drawing.draw_detection(img_final, detection)
		
		# Show repeted frames (Video)
		cv2.imshow('Face Detection Mediapipe', img_final)
		cv2.waitKey(0)
cv2.destroyAllWindows()

detectFromImg('./1.jpg')