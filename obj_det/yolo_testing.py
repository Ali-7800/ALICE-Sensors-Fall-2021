
import cv2
import numpy as np
import argparse
import time


import RPi.GPIO as GPIO
from os import listdir
from os.path import isfile, join


GPIO.setmode(GPIO.BOARD)
left = 3
right = 5

GPIO.setup(left, GPIO.OUT)
GPIO.setup(right, GPIO.OUT)

last_notif = 0
last_left_notif=0
last_right_notif=0
#This exec line streams from the realsense
#exec(open("RealSenseStreaming.py").read())


parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--test', help="True/False", default=False)
parser.add_argument('--play_video', help="Tue/False", default=False)
parser.add_argument('--image', help="Tue/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default="/home/pi/ME470/obj_det/videos/car_on_road.mp4")
parser.add_argument('--image_path', help="Path of image to detect objects", default="/home/pi/ME470/obj_det/Images/bicycle.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()



#Load yolo
def load_yolo():
	netL = cv2.dnn.readNet("/home/pi/ME470/obj_det/Light Weights and Config/custom-yolov4-tiny-detector_best.weights", "/home/pi/ME470/obj_det/Light Weights and Config/custom-yolov4-tiny-detector.cfg")
	netL.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	netL.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	netD = cv2.dnn.readNet("/home/pi/ME470/obj_det/Dark Weights and Config/custom-yolov4-tiny-detector_best.weights", "/home/pi/ME470/obj_det/Dark Weights and Config/custom-yolov4-tiny-detector.cfg")
	netD.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	netD.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	classes = []
	with open("/home/pi/ME470/obj_det/_classes_t1.txt", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layers_namesL = netL.getLayerNames()
	output_layersL = netL.getUnconnectedOutLayersNames()
	layers_namesD = netD.getLayerNames()
	output_layersD = netD.getUnconnectedOutLayersNames()
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return netL, netD, classes, colors, output_layersL, output_layersD



def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.2, fy=0.2)
	value = np.mean(img)
	height, width, channels = img.shape
	return img, height, width, channels, value

def start_webcam():
	cap = cv2.VideoCapture(0)
    #exec(open("RealSenseStreaming.py").read())

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



def send_notif(x,y,w,h,height,width):
	global last_notif
	global last_left_notif
	global last_right_notif
	#print(x)
	#print(y)
	leftwheel=width/2-100
	rightwheel=width/2
	m = (354-240)/(149-320)
	xc = x+w/2
	yc = y+h/2
	xl = x
	yl = y + h
	xr = x + w
	yr = yl
	if (((xl<320 and xr>320) and yl>240)or((yr-240)>m*(xr-320) and xr<320)):
	
		if((yc-240)>m*(xc-320) and xc<320):
			if time.time()-last_notif>2:
				print('Hazard')
				GPIO.output(left, GPIO.HIGH)
				GPIO.output(right, GPIO.HIGH)
				time.sleep(0.01)
				GPIO.output(left, GPIO.LOW)
				GPIO.output(right, GPIO.LOW)
				last_notif=time.time()
			#print('time.time()',time.time())
		elif(xc<320):
			if time.time()-last_left_notif>2:
				print('Hazard Left')
				GPIO.output(left, GPIO.HIGH)
				time.sleep(0.01)
				GPIO.output(left, GPIO.LOW)
				last_left_notif=time.time()
			#print('time.time()',time.time())
		else:
			if time.time()-last_right_notif>2:
				print('Hazard Right')
				GPIO.output(right, GPIO.HIGH)
				time.sleep(0.01)
				GPIO.output(right, GPIO.LOW)
				last_right_notif=time.time()
		#print('time.time()',time.time())
	else:
	    print("No Hazard Detected")
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
				print(detect[1])
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
				#if center_y > height/2:
					#if time.time()-last_notif>2:
						#send_notif(x,y,w,h,height,width)
	return boxes, confs, class_ids



def draw_labels(boxes, confs, colors, class_ids, classes, img, height, width):
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.15, 0.1)
    #indexes = cv2.dnn.boxes(boxes,confs)
	font = cv2.FONT_HERSHEY_PLAIN
    #print(confs)
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			center_y = int(y + h/2)
			center_x = int(x + w/2)
			#height = 605
			#width = 806
			#print(center_y)
			#print(height)
			print(center_x,center_y)
			if center_y > 480 / 4:
				send_notif(x, y, w, h, 480, 640)
			label = str(classes[class_ids[i]]) + " " + str(confs[i])
			color = [255, 0, 0]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
			
	return img
		
		
	#cv2.imwrite(output, img)

def image_detect(img_path):
	modelL, modelD, classes, colors, output_layersL,output_layersD = load_yolo()
	image, height, width, channels, value = load_image(img_path)
	if value<110:
		blob, outputs = detect_objects(image, modelD, output_layersD)
		print("Dark")
	else:
		blob, outputs = detect_objects(image, modelL, output_layersL)
		print("Light")
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width,classes)
	draw_labels(boxes, confs, colors, class_ids, classes, image, height, width)
	#while True:
	#	key = cv2.waitKey(1)
	#	if key == 27:
	#		break

def webcam_detect():
	netL, netD, classes, colors, output_layersL, output_layersD = load_yolo()
	cap = start_webcam()
	while True:
		for i in range(12):
			_, frame = cap.read()
		height, width, channels = frame.shape
		value = np.mean(frame)
		if value<110:
		    blob, outputs = detect_objects(frame, netD, output_layersD)
		    print("Dark")
		else:
		    blob, outputs = detect_objects(frame, netL, output_layersL)
		    print("Light")
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width,classes)
		frame = draw_labels(boxes, confs, colors, class_ids, classes, frame, height, width)
		d = 10
		frame = cv2.line(frame, (320,240), (144,480), (0, 255, 0), 2)
		frame = cv2.line(frame, (320,240), (320,480), (0, 0, 255), 2)
		frame = cv2.line(frame, (320,240), (0,453), (0, 0, 255), 2)
		frame = cv2.resize(frame, (1280, 960))
		cv2.imshow("feed",frame)
		cv2.waitKey(10)
	cap.release()



def start_video(video_path):
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		print(frame.shape)
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
	test = args.test
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
	if test:
		mypath = "/home/me470/testing_images/set3"
		image_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
		for i in image_names:
			image_path = mypath +"/"+ i
			output = mypath +"_t3/"+ i
			print(enumerate(i))
			image_detect(image_path,output)

	
	cv2.destroyAllWindows()
