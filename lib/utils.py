import torch
import time
import numpy as np
from torchvision.ops import nms
import cv2

def load_model(Model, path, device):
    data = torch.load(path)
    state_dict = data['state_dict']
    model = Model(**data['kwargs'])
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

def save_model(model, path, kwargs):
    stamp = str(int(time.time()))
    data = {'state_dict': model.state_dict(), "kwargs": kwargs}
    torch.save(data, path + '{}.pth'.format(stamp))
    print("Saved model with stamp {}.".format(stamp))

def get_boxes(prediction, obj_thr, batch_size, img_dim, grid_dim, iou_thr=0.5):
    boxes = []
    for batch in range(batch_size):
        yy, xx = torch.where(prediction[batch,0,:,:] > obj_thr)
        #print(yy, xx)
        boxes_batch = torch.empty((yy.size()[0], 4))
        scores_batch = torch.empty(yy.size()[0])

        for i in range(yy.size()[0]):
            yr, xr, w, h = prediction[batch, 1:5, yy[i], xx[i]]
            w = img_dim[1] * w / 2
            h = img_dim[0] * h / 2
            aux_x = (xx[i] + xr) * grid_dim[1]
            aux_y = (yy[i] + yr) * grid_dim[0]
            x1 = int(aux_x - w)
            x2 = int(aux_x + w)
            y1 = int(aux_y - h)
            y2 = int(aux_y + h)
            boxes_batch[i,:] = torch.tensor([x1, y1, x2, y2])
            scores_batch[i] = prediction[batch, 0, yy[i], xx[i]]
        
        boxes_indices = nms(boxes_batch, scores_batch, iou_thr)
        boxes.append(boxes_batch[boxes_indices])
    
    return boxes

def draw_boxes(image, boxes, box_type='prediction'):
    for box in boxes:
        x1, y1, x2, y2 = box.numpy()
        if box_type == 'prediction':
            color = (255,0,0, 255)
            thickness = 2
        else:
            color = (0,255,0, 255)
            thickness = 3
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


