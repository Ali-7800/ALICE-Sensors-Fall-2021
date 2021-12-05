
import cv2
import numpy as np
import argparse
import time
import os
#import slack_notifications as slack

#These varialbes are used to space out notifications
last_notif = 0
last_left_notif=0
last_right_notif=0
#This exec line streams from the realsense
#exec(open("RealSenseStreaming.py").read())

#select the input
parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--play_video', help="Tue/False", default=False)
parser.add_argument('--image', help="Tue/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default="videos/car_on_road.mp4")
parser.add_argument('--image_path', help="Path of image to detect objects", default="Images/bicycle.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()


#Load yolo
def load_yolo():
	net = cv2.dnn.readNet("custom-yolov4-tiny-detector_3000.weights", "custom-yolov4-tiny-detector.cfg")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	classes = []
	with open("_classes.txt", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layers_names = net.getLayerNames()
	output_layers = net.getUnconnectedOutLayersNames()
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers



#used if we have photo as input
def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.2, fy=0.2)
	height, width, channels = img.shape
	return img, height, width, channels

#Used if using webcam
def start_webcam():
	cap = cv2.VideoCapture(0) #Change this number to proper input
	return cap


def display_blob(blob):
	'''
		Three images each for RED, GREEN, BLUE channel
	'''
	for b in blob:
		for n, imgb in enumerate(b):
			cv2.imshow(str(n), imgb)


def detect_objects(img, net, outputLayers):
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

#This function sends the notifications
def send_notif(x,y,w,h,height,width):
	global last_notif
	global last_left_notif
	global last_right_notif
	leftwheel=width/2-y/1 #path traced by left wheel
	rightwheel=width/2 #path traced by right wheel (assume mounted over right wheel)
	if(x < leftwheel+width/10 and x+w>rightwheel-width/10): #if hazard spans the whole width
		if time.time()-last_notif>2: #Only send notification if its been more than 2 seconds since same notification
			print('Hazard')
			slack.send_notify('general', username='--HAZARD--', text='HAZARD')
			last_notif=time.time() #update last notification
	if( x < leftwheel+width/10 and x+w > leftwheel-width/10): #if hazard i son path of left wheel
		if time.time()-last_left_notif>2: #Only send notification if its been more than 2 seconds since same notification
			print('Hazard Left')
			slack.send_notify('general', username='--HAZARD LEFT--', text='HAZARD LEFT')
			last_left_notif=time.time() #update last notification
	if( x < rightwheel+width/10 and x+w> rightwheel-width/10): #if hazard on path of right wheel
		if time.time()-last_right_notif>2: #Only send notification if its been more than 2 seconds since same notification
			print('Hazard Right')
			slack.send_notify('general', username='--HAZARD RIGHT--', text='HAZARD RIGHT')
			last_right_notif=time.time() #update last notification
	return last_notif

def get_box_dimensions(outputs, height, width,classes):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0:
				print(str(classes[class_id]) + " " + str(conf))
			if conf > .1:
				center_x = int(detect[0] * width) #center of hazard in pixels x direction
				center_y = int(detect[1] * height) #center of hazard in pixels y direction
				w = int(detect[2] * width) #width of hazard in pixels
				h = int(detect[3] * height) #height of hazard in pixels
				x = int(center_x - w/2) #left edge
				y = int(center_y - h / 2) #top edge
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids



def draw_labels(boxes, confs, colors, class_ids, classes, img, height, width):
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.15, 0.1)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			center_y = int(y + h/2) #center of hazard in pixels
			#if center_y > 480 / 2 and center_y<(7*480/8): #if in bottom half of frame but not in bottom 1/8
				#send_notif(x, y, w, h, 480, 640) #Call notification function
			label = str(classes[class_ids[i]]) + " " + str(confs[i]) #Displays on display
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	cv2.imshow("Image", img)

def image_detect(img_path):
	model, classes, colors, output_layers = load_yolo()
	image, height, width, channels = load_image(img_path)
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width,classes)
	draw_labels(boxes, confs, colors, class_ids, classes, image, height, width)
	while True:
		key = cv2.waitKey(1)
		if key == 27:
			break

def webcam_detect():
	model, classes, colors, output_layers = load_yolo()
	cap = start_webcam()
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width,classes)
		draw_labels(boxes, confs, colors, class_ids, classes, frame, height, width)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()



def start_video(video_path):
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width,classes)
		draw_labels(boxes, confs, colors, class_ids, classes, frame, height, width)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()


if __name__ == '__main__':
	webcam = args.webcam
	video_play = args.play_video
	image = args.image
	if webcam:
		if args.verbose:
			print('---- Starting Web Cam object detection ----')
		webcam_detect()
	if video_play:
		video_path = args.video_path
		if args.verbose:
			print('Opening '+video_path+" .... ")
		start_video(video_path)
	if image:
		image_path = args.image_path
		if args.verbose:
			print("Opening "+image_path+" .... ")
		image_detect(image_path)


	cv2.destroyAllWindows()
