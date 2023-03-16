from pytorch_model_summary import summary
from detect_base.models.rhnet import rHnet
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using device:", device)

model = rHnet(stem_dims=[64, 64], depths=[4, 4, 8, 8], dims=[64, 128, 256, 512])
#model = rHnet()
model.eval()
model.to(device)
input = torch.rand((1, 3, 512, 512)).cuda()
print(summary(model, torch.zeros((1,3,512,512)).cuda(), show_input=False, show_hierarchical=False))
#exit()
predict = model(input)

print(predict.shape)

batch_sizes = np.arange(16,129,16)
batch_size = 64
sizes = np.arange(32,513,8)

times = []

for size in sizes:
    input = torch.rand((batch_size, 3, 512, 512)).cuda()

    start_time = float(time.time())
    out = model(input)
    end_time = float(time.time())

    print(size, ':', end_time-start_time)
    times.append(end_time-start_time)


print("mean:", np.mean(times))
plt.plot(sizes, times)
plt.show()


