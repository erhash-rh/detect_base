from models.loss import detectLoss
import torch

outputs = torch.rand(3,5,6,7)
targets = torch.rand(3,5,6,7)


obj = targets[:,0,:,:]
noobj = torch.abs(obj-1)

# Objectiveness loss
aux = (outputs[:,0,:,:] - obj).pow(2)
loss_obj = torch.mean(obj*aux)
loss_noobj = torch.mean(noobj*aux)

# Position loss
aux2 = (outputs[:,1:3,:,:] - targets[:,1:3,:,:]).pow(2)
print('aux2', aux2.size())
aux = torch.mean((outputs[:,1:3,:,:] - targets[:,1:3,:,:]).pow(2), dim=1)
print('pos', aux.size())
loss_pos = torch.mean(obj * torch.mean((outputs[:,1:3,:,:] - targets[:,1:3,:,:]).pow(2), dim=1))

# Box loss

loss_box = torch.mean(obj * torch.mean((outputs[:,3:5,:,:] - targets[:,3:5,:,:]).pow(2), dim=1))

loss = loss_obj + loss_noobj + loss_pos + loss_box

