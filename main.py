import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from lib.utils import save_model, get_boxes, draw_boxes

from models.rhnet import rHnet, rHnetFPN
from models.loss import detectLoss
from lib.dataset import load_dataset
import time
import cv2

def main():
	batch_size = 32
	epochs = 16
	lr = 1e-3
	weight_decay = 1e-5
	lambda_lr = lambda epoch: 0.8**epoch
	save_path = './models/'
	tt_split = 0.9

	img_dir = 'D:/py/vaas/Datasets/WiderPerson/imgs-cut/'
	label_dir = 'D:/py/vaas/Datasets/WiderPerson/labels-cut/'
	#img_dir = 'D:/py/vaas/KanjiAnki-git/imgs/'
	#label_dir = 'D:/py/vaas/KanjiAnki-git/labels/'
	
	save_path = './models/save/'

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print("Training on {}.".format(device))

	trainloader, testloader = load_dataset(img_dir, label_dir, split=tt_split, batch_size=batch_size)
	kwargs = {'stem_dims': [32], 'depths': [8,8,8], 'dims': [64, 128, 256]}
	model = rHnet(**kwargs)
	model.to(device)


	train(model, device, trainloader, detectLoss, epochs, lr, lambda_lr, weight_decay=weight_decay)
	save_model(model, save_path, kwargs)
	view_prediction(model, device, testloader)


def train(model, device, trainloader, criterion, epochs, lr, lambda_lr, weight_decay = 1e-5):
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

	for epoch in range(epochs):  # loop over the dataset multiple times
		with tqdm(trainloader, unit='batch') as tepoch:
			loss = torch.tensor(0)
			last_loss = 0
			for data in tepoch:
				last_loss = last_loss*0.9+loss.item()*0.1
				tepoch.set_description(f'Epoch {epoch} | loss={"%.5f" % round(last_loss, 5)}')
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data[0].to(device), data[1].to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
		
		scheduler.step()

def view_prediction(model, device, testloader, n_images=8, obj_thr=0.7, img_dim=(512,512), grid_dim=(16,16), iou_thr=0.8):
	data = next(iter(testloader))
	images, labels_truth = data[0].to(device), data[1]
	labels_prediction = model(images)
	images = np.ascontiguousarray(torch.permute(images, (0,2,3,1)).cpu().numpy() * 255, dtype=np.uint8)
	boxes_truth = get_boxes(labels_truth, obj_thr, n_images, img_dim, grid_dim, iou_thr)
	boxes_prediction = get_boxes(labels_prediction, obj_thr, n_images, img_dim, grid_dim, iou_thr)
	for i in range(n_images):
		image = images[i]
		draw_boxes(image, boxes_truth[i], box_type='truth')
		draw_boxes(image, boxes_prediction[i], box_type='prediction')	
		cv2.imshow(str(i), image)
	cv2.waitKey(0)
		

def test(model, device, testloader):
	correct = 0
	total = 0
	# since we're not training, we don't need to calculate the gradients for our outputs
	with torch.no_grad():
		for data in testloader:
			images, labels = data[0].to(device), data[1].to(device)
			# calculate outputs by running images through the network
			outputs = model(images)
			# the class with the highest energy is what we choose as prediction
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


if __name__ == '__main__':
	main()