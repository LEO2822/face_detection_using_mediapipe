# Importing modules
import cv2
import mediapipe as mp

# Initialize mediapipe face detector and drawer
mp_face_detection = mp.solutions.face_detection
mp_face_drawing   = mp.solutions.drawing_utils

# Detect faces on video or cam
def detectFromVid(video_path=None, cam=False):
	"""Function to detect faces on a video or cam"""

	# Using mediapipe face detector
	with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:

		# Load the video or cam based on cam flag
		if cam:
			video = cv2.VideoCapture(0)
		else:
			video = cv2.VideoCapture(video_path)

		# Loop over video frames
		while True:
		
            # Get every frame on the video
			success, frame = video.read()

			# If there's no more frames, stop 
			if not success:
				break

			# Convert the BGR frame to RGB
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# Get faces on the an image (frame)
			faces = face_detection.process(frame)

			# Convert the frame back to its colors for displaying
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

			# Draw the face detection annotations on the image
			if faces.detections:
				for detection in faces.detections:
					mp_face_drawing.draw_detection(frame, detection)
			
			# Show repeted images (Video)
			cv2.imshow('Face Detection Mediapipe', frame)
			if cv2.waitKey(5) & 0xFF == ord('q'):
				break
		# End 
	video.release()

detectFromVid('./video.mp4' , False)