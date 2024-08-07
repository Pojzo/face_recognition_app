import cv2

class Camera: 
	def __init__(self):
		self.cap = cv2.VideoCapture(0)
		self.camera_open = False
		if not self.cap.isOpened():
			raise Exception("Camera failed to open")
		
		self.camera_open = True
	
	def get_frame(self):
		if not self.camera_open:
			return None
	
		_, frame = self.cap.read()
		return frame
	
	def destroy(self):
		self.cap.release()
