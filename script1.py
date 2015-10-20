import cv2, os, argparse, numpy

DEFAULT_OUTPUT_PATH = 'FaceCaptureImages/'
DEFAULT_CASCADE_INPUT_PATH = 'haarcascade_frontalface_alt.xml'

class VideoCapture(object):
	
	def __init__(self):
		self.count = 0
		self.argsObj = Parse()
		self.faceCascade = cv2.CascadeClassifier(self.argsObj.input_path)
		self.videoSource = cv2.VideoCapture(0)

	def CaptureFrames(self):
		while True:
			# Create a unique no for each frame
			frameNumber = '%08d' % (self.count)

			# Cpature frame by frame
			ret, frame = self.videoSource.read()

			# set screen color to grey, so the haar cascade can easily detect the edges and faces
			screenColor = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Customize how the cascade detects your face
			faces = self.faceCascade.detectMultiScale(
				screenColor, 
				scaleFactor = 1.1, 
				minNeighbors = 5, 
				minSize = (30,30), 
				flags = cv2.CASCADE_SCALE_IMAGE)

			# Display the resulting frame
			cv2.imshow('Video Frames', screenColor)

			# If length(no) of faces is zero , no faces detected
			if len(faces) ==  0:
				pass

			# if a face is detected, faces return >= 1
			elif len(faces) > 0:
				print('Face Detected')

				# Graph the face and draw rectangle aroung it
				for (x,y,w,h) in faces:
					cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

				cv2.imwrite(DEFAULT_OUTPUT_PATH + frameNumber + '.png', frame)

			# Increment count so we get a unique name for each frame
			self.count += 1

			# If 'esc' is hit the video is closed. We are going to wait a fraction of a second per loop
			if cv2.waitKey(1) == 27:
				break


		# when everything is done , release the capture and close windows
		self.videoSource.release()
		cv2.waitKey(500)
		cv2.destroyAllWindows()
		cv2.waitKey(500)

def Parse():
	parser = argparse.ArgumentParser(description='Cascade Path for face detection')
	parser.add_argument('-i', '--input_path', type = str, default = DEFAULT_CASCADE_INPUT_PATH, help='Cascade input path')
	parser.add_argument('-o', '--output_path', type = str, default = DEFAULT_OUTPUT_PATH, help = 'Output Path for pictures taken')
	args = parser.parse_args()
	return args


def ClearImageFolder():
	if not os.path.exists(DEFAULT_OUTPUT_PATH):
		os.makedirs(DEFAULT_OUTPUT_PATH)

	else:
		for files in os.listdir(DEFAULT_OUTPUT_PATH):
			filePath = os.path.join(DEFAULT_OUTPUT_PATH, files)
			if os.path.isfile(filePath):
				os.unlink(filePath)
			else:
				continue

def main():
	ClearImageFolder()

	# Instatiate Class Object
	faceDetectImplementation = VideoCapture()

	# Call CaptureFrames from Class to start face detection
	faceDetectImplementation.CaptureFrames()

if __name__ == '__main__':
	main()