import numpy as np
import pickle as pkl
import time
import cv2
import sys
import torchvision.transforms as transforms
import torch

from lib.utils import save_model, get_boxes, draw_boxes, load_model
from models.rhnet import rHnet
from models.loss import detectLoss
from lib.dataset import load_dataset
import time
import cv2

################################### Paths
model_dir = './models/save/'
model_name = '1667277156.pth'

video_dir = 'D:/py/vaas/Pedestrians/videos/'
video_name = '4.mp4'

classes = ['car', 'van', 'bus']
colors = [(0,255,0),(255,0,0),  (0,0,255)]

################################### Load model
print("Loading model...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = load_model(rHnet, model_dir+model_name, device)
print("Done")

################################### Load video
cap = cv2.VideoCapture(video_dir + video_name)
ret, frame = cap.read()

################################### Inference parameters
iou_thr = 0.5
obj_thr = 0.5
class_thr = 0.5
display_h = 640


USE_TRACKER = False
USE_PROJECTOR = False
det_thr = 1 # tracker detection threshold - consecutive frames
dist_thr = 0.03 # needs to be bigger for generally faster objects

################################### Build tracker and projector
if USE_TRACKER:
	tracker = Tracker(input_h, input_w, dist_thr=dist_thr, lookback=10, yroi=yroi)
if USE_PROJECTOR:
	projector = Projector(1, 0.2, input_h)

################################### Run
transform1=transforms.ToTensor()
transform2=transforms.CenterCrop(size=(512,512))
image = transform1(frame)
c, h, w = image.size()
print(image.size())
#exit()
if h > w:
	new_size = int(h*(512/w), 512)
else:
	new_size = (512, int(w*(512/h)))
transform3=transforms.Resize(size=new_size)


while (cap.isOpened()):
	ret, frame = cap.read()
	# Cut and resize frame
	image = transform1(frame)
	c, h, w = image.size()
	#exit()
	image = transform3(image)
	image = transform2(image)
	image = image.reshape((1, *image.size()))
	# Predict and NMS
	prediction = model(image.cuda())
	boxes = get_boxes(prediction, obj_thr, 1, (512,512), (16,16))
	#print(boxes[0])
	images = np.ascontiguousarray(torch.permute(image, (0,2,3,1)).cpu().numpy() * 255, dtype=np.uint8)
	#print(images.shape)
	draw_boxes(images[0], boxes[0])

	#print(prediction.size())

	cv2.imshow("inference", images[0])

	k = cv2.waitKey(20) & 0xFF
	if k == 27:
		break
	elif k == ord('a'):
		obj_thr-=0.01
		print(np.round(obj_thr,2))
	elif k == ord('s'):
		obj_thr+=0.01
		print(np.round(obj_thr,2))


cap.release()
cv2.destroyAllWindows()

