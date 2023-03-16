from pytorch_model_summary import summary
from models.rhnet import rHnet
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from lib.utils import load_model, get_boxes, draw_boxes
from lib.dataset import load_dataset
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using device:", device)

model_path = './models/save/'
file_name = '1666789158.pth'

model = load_model(rHnet, model_path+file_name, device, depths=[4,8], dims=[128,256])

batch_size = 16
input = torch.rand((batch_size, 3, 512, 512)).cuda()
#print(summary(model, torch.zeros((1,3,512,512)).cuda(), show_input=False, show_hierarchical=False))
#exit()
prediction = model(input)

print(prediction.shape)


img_dir = 'D:/py/vaas/Datasets/WiderPerson/imgs-cut/'
label_dir = 'D:/py/vaas/Datasets/WiderPerson/labels-cut/'
save_path = './models/save/'

trainloader, testloader = load_dataset(img_dir, label_dir, split=0.9, batch_size=batch_size)

test_on = 'inference'

for data in testloader:
    images, labels = data[0].to(device), data[1]
    if test_on == 'inference':
        labels = model(images)
    images = torch.permute(images, (0,2,3,1)).cpu().numpy() * 255
    for i, image in enumerate(images):
        image = np.ascontiguousarray(image, dtype=np.uint8)
        boxes = get_boxes(labels, 0.05, batch_size, (512,512), (16,16), iou_thr = 0.9)
        draw_boxes(image, boxes[i])
        cv2.imshow(str(i), image)
    cv2.waitKey(0)
    exit()


    

print(len(boxes))
print(boxes)





