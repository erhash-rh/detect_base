import torch
import torchvision
import torchvision.transforms as transforms
from os import listdir
from os.path import isfile, join
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np

class DetectionDataset(Dataset):

    def __init__(self, img_dir, label_dir, n_classes=0, dim=[32,32], img_dim=(512,512), transform=transforms.ToTensor()):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.n_classes = n_classes
        self.dim = dim
        self.img_dim = img_dim
        self.dx = img_dim[1]/dim[1]
        self.dy = img_dim[0]/dim[0]

        self.img_files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
        self.label_files = [f for f in listdir(label_dir) if isfile(join(label_dir, f))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        sample = self.transform(cv2.imread(self.img_dir + self.img_files[idx]))
        labels = self.transform(self.get_labels(idx))
        return sample, labels
    
    def get_labels(self, idx):
        target = np.zeros((self.dim[0], self.dim[1], 5+self.n_classes)) # ToTensor permutes the data to C, H, W
        with open(self.label_dir + self.label_files[idx], 'r') as f:
            for line in f.readlines():
                class_idx, xr, yr, wr, hr = [float(x) for x in line.split(' ')]
                xx = xr * self.img_dim[1]
                yy = yr * self.img_dim[0]
                x = int(xx / self.dx)
                y = int(yy / self.dy)
                xpos = xx % self.dx / self.dx
                ypos = yy % self.dy / self.dy
                target[y, x, :5] = np.asarray([1, ypos, xpos, wr, hr])
                if self.n_classes != 0:
                    target[y, x, 5+int(class_idx)] = 1
        return target

def load_dataset(img_dir, label_dir, n_classes=0, dim=[32,32], split=0.8, batch_size=32):
    full_dataset = DetectionDataset(img_dir, label_dir, n_classes=n_classes, dim=dim)
    train_size = int(split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def load_dataset_base(batch_size):
	transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='d:/py/vaas/Datasets/', train=True,
											download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='d:/py/vaas/Datasets/', train=False,
										download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
											shuffle=False, num_workers=2)
	classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	
	return trainloader, testloader, classes