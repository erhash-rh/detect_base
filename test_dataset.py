from detect_base.lib.dataset import DetectionDataset, load_dataset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

img_dir = 'D:/py/vaas/Datasets/WiderPerson/imgs-cut/'
label_dir = 'D:/py/vaas/Datasets/WiderPerson/labels-cut/'

transform = transforms.Compose([transforms.ToTensor()])

dataset = DetectionDataset(img_dir, label_dir, transform = transform)

item = dataset[0]

train_dataloader, test_dataloader = load_dataset(img_dir, label_dir, split=0.8)
out = next(iter(train_dataloader))

print(len(train_dataloader)*32)
print(len(test_dataloader)*32)

