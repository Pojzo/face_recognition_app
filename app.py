from camera import Camera

import time
from tkinter import Frame, Label
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import face_recognition as fr

class App:
	def __init__(self, root):
		self.root = root
		self.root.title("Video Feed App")
		self.root.geometry("800x600")  # Set the window size to 800x600

		self.camera = Camera()

		self.is_running = True
		self.size_configured = False
		self.detecting_faces = True

		self.create_ui()
		self.show_frame()

	def create_ui(self):
		self.button_frame = Frame(self.root)
		self.button_frame.pack(side=tk.LEFT, padx=10)

		self.button1 = tk.Button(self.button_frame, command=self.dummy_action)
		self.button1.pack(pady=10)

		self.button2 = tk.Button(self.button_frame, command=self.dummy_action)
		self.button2.pack(pady=10)

		# Label for video feed
		self.video_label = Label(self.root)
		self.video_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

		self.detect_faces_var = tk.BooleanVar(value=True)
		self.detect_landmarks_var = tk.BooleanVar(value=True)

		self.detect_faces_checkbox = tk.Checkbutton(
			self.button_frame,
			text="Detect faces",
			variable=self.detect_faces_var,
			command=self.is_detecting_faces
		)

		self.detect_landmarks_checkbox = tk.Checkbutton(
			self.button_frame,
			text="Detect landmarks",
			variable=self.detect_landmarks_var,
			command=self.is_detecting_landmarks
		)

		self.detect_faces_checkbox.pack(pady=10)
		self.detect_landmarks_checkbox.pack(pady=10)

		self.fps_text = tk.Text(self.root, height=10, width=20)
		
		self.fps_text.pack(side=tk.TOP, anchor=tk.NE)
	
	def get_frame(self, size_reduction=0.6):
		frame = self.camera.get_frame()
		if frame is None:
			return None

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		new_width = frame.shape[1] * size_reduction
		new_height = frame.shape[0] * size_reduction

		frame = cv2.resize(frame, (int(new_width), int(new_height)))

		return frame
	
	def ensure_correct_size(self):
		if not self.size_configured:
			self.root.geometry(f"{self.frame.shape[1]}x{self.frame.shape[0]}")

	def show_frame(self):
		start_time = time.time()
		if not self.is_running:
			return

		self.frame = self.get_frame()

		if self.is_detecting_faces():
			face = self.detect_face(self.frame)
			if not face is None:
				for (top, right, bottom, left) in face:
					cv2.rectangle(self.frame, (left, top), (right, bottom), (0, 255, 0), 2)
			
		if self.is_detecting_landmarks() and self.is_detecting_faces():
			landmarks = self.detect_landmarks(self.frame, face)
			for landmark in landmarks:
				for (x, y) in landmark:
					pass
					cv2.circle(self.frame, (x, y), 2, (0, 0, 255), -1)  # Draw landmarks in red


		end_time = time.time()
		fps = 1 / (end_time - start_time)

		cv2.putText(self.frame, f"FPS: {fps:.2f}", (60, self.frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

		img = Image.fromarray(self.frame)
		imgtk = ImageTk.PhotoImage(image=img)

		self.video_label.imgtk = imgtk
		self.video_label.configure(image=imgtk)

		self.ensure_correct_size()
		self.root.after(10, self.show_frame)

	def detect_face(self, frame, fx=0.25, fy=0.25):
		downsampled_frame = cv2.resize(frame, (0, 0), fx=fx, fy=fy)
		
		face_locations = fr.face_locations(downsampled_frame)
		
		if not len(face_locations):
			return None
		
		adjusted_face_locations = []
		for (top, right, bottom, left) in face_locations:
			# Calculate the adjusted coordinates
			adjusted_top = int(top / fy)
			adjusted_right = int(right / fx)
			adjusted_bottom = int(bottom / fy)
			adjusted_left = int(left / fx)
			adjusted_face_locations.append((adjusted_top, adjusted_right, adjusted_bottom, adjusted_left))
		
		return adjusted_face_locations
	
	def detect_landmarks(self, face_image, face_locations=None, model="large"):
		landmarks_dict = fr.face_landmarks(face_image, face_locations, model)
		
		# Convert the landmarks from dicts to an array of tuples
		landmarks_as_array = []
		
		for landmark in landmarks_dict:
			face_landmarks_array = []
			
			# Extract coordinates from the dictionary and append them to the list
			for key in landmark:
				points = landmark[key]
				face_landmarks_array.extend(points)  # Flatten the list of points into a single list
				
			landmarks_as_array.append(face_landmarks_array)  # Add to the main array

		return landmarks_as_array	
	def is_detecting_faces(self):
		return self.detect_faces_var.get()

	def is_detecting_landmarks(self):
		return self.detect_landmarks_var.get()
	
	def dummy_action(self):
		# Placeholder for button actions
		pass

	def on_closing(self):
		self.camera.destroy()
		self.root.destroy()