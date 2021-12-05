
import cv2
import numpy as np
import argparse
import time
import slack_notifications as slack
slack.ACCESS_TOKEN = 'xoxb-1971148086049-1958754772211-f7MbTJ2QeZR7fGZEBJX3VIvC'
from os import listdir
from os.path import isfile, join
mypath = "/home/me470/testing_images/set3"
image_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

last_notif = 0
last_left_notif=0
last_right_notif=0
#This exec line streams from the realsense
#exec(open("RealSenseStreaming.py").read())


#parser = argparse.ArgumentParser()
#parser.add_argument('--webcam', help="True/False", default=False)
#parser.add_argument('--play_video', help="Tue/False", default=False)
#parser.add_argument('--image', help="Tue/False", default=False)
#parser.add_argument('--video_path', help="Path of video file", default="videos/car_on_road.mp4")
#parser.add_argument('--image_path', help="Path of image to detect objects", default="Images/bicycle.jpg")
#parser.add_argument('--verbose', help="To print statements", default=True)
#rgs = parser.parse_args()



#Load yolo
def load_yolo():
	net = cv2.dnn.readNet("/home/me470/testing_images/t2_weights/custom-yolov4-tiny-detector_best.weights", "/home/me470/testing_images/t2_weights/custom-yolov4-tiny-detector.cfg")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	classes = []
	with open("_classes_t1.txt", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers

''' #something taken from a keras example
def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))
    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))
    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))
    return Model(inputs, [y1,y2,y3])
	'''

def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.2, fy=0.2)
	height, width, channels = img.shape
	return img, height, width, channels

def start_webcam():
	cap = cv2.VideoCapture(4)
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

''' old notifications code slack
def send_notif(x,y,w,h,height,width):
	leftwheel=width/2-y/1
	rightwheel=width/2
	if (x<leftwheel+width/10 and x+w>rightwheel-width/10):
		slack.send_notify('general', username='--HAZARD--', text='HAZARD')
	elif (x<leftwheel+width/10 and x+w>leftwheel-width/10):
		slack.send_notify('general', username='--HAZARD LEFT--', text='HAZARD LEFT')
	elif (x<rightwheel+width/10 and x+w>rightwheel-width/10):
		slack.send_notify('general', username='--HAZARD RIGHT--', text='HAZARD RIGHT')
'''

def send_notif(x,y,w,h,height,width):
	global last_notif
	global last_left_notif
	global last_right_notif
	#print(x)
	#print(y)
	leftwheel=width/2-y/1
	rightwheel=width/2
	if(x < leftwheel+width/10 and x+w>rightwheel-width/10) or ((x < leftwheel+width/10 and x+w > leftwheel-width/10) and ( x < rightwheel+width/10 and x+w> rightwheel-width/10)):
		if time.time()-last_notif>2:
			print('Hazard')
			slack.send_notify('general', username='--HAZARD--', text='HAZARD')
			last_notif=time.time()
		#print('time.time()',time.time())
	if( x < leftwheel+width/10 and x+w > leftwheel-width/10):
		if time.time()-last_left_notif>2:
			print('Hazard Left')
			slack.send_notify('general', username='--HAZARD LEFT--', text='HAZARD LEFT')
			last_left_notif=time.time()
		#print('time.time()',time.time())
	if( x < rightwheel+width/10 and x+w> rightwheel-width/10):
		if time.time()-last_right_notif>2:
			print('Hazard Right')
			slack.send_notify('general', username='--HAZARD RIGHT--', text='HAZARD RIGHT')
			last_right_notif=time.time()
		#print('time.time()',time.time())
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



def draw_labels(boxes, confs, colors, class_ids, classes, img, height, width,output):
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.15, 0.1)
    #indexes = cv2.dnn.boxes(boxes,confs)
	font = cv2.FONT_HERSHEY_PLAIN
    #print(confs)
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			center_y = int(y + h/2)
			#height = 605
			#width = 806
			#print(center_y)
			#print(height)
			print(center_y)
			#if center_y > 480 / 2 and center_y<(7*480/8):
			#	send_notif(x, y, w, h, 480, 640)
			label = str(classes[class_ids[i]]) + " " + str(confs[i])
			color = [255, 0, 0]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	cv2.imwrite(output, img)

def image_detect(img_path,output):
	model, classes, colors, output_layers = load_yolo()
	image, height, width, channels = load_image(img_path)
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width,classes)
	draw_labels(boxes, confs, colors, class_ids, classes, image, height, width,output)
	#while True:
	#	key = cv2.waitKey(1)
	#	if key == 27:
	#		break

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
		print(frame.shape)
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width,classes)
		draw_labels(boxes, confs, colors, class_ids, classes, frame, height, width)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()


if __name__ == '__main__':
	for i in image_names:
		image_path = mypath +"/"+ i
		output = mypath +"_t2/"+ i
		print(enumerate(i))
		image_detect(image_path,output)


	cv2.destroyAllWindows()
