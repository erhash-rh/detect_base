import torch
import torch.nn as nn

def detectLoss(outputs, targets):
    
    obj = targets[:,0,:,:]
    noobj = torch.abs(obj-1)

    # Objectiveness loss
    aux = (outputs[:,0,:,:] - obj).pow(2)
    loss_obj = torch.mean(obj*aux)
    loss_noobj = torch.mean(noobj*aux)

    # Position loss
    loss_pos = torch.mean(obj * torch.mean((outputs[:,1:3,:,:] - targets[:,1:3,:,:]).pow(2), dim=1))

    # Box loss
    loss_box = torch.mean(obj * torch.mean((outputs[:,3:5,:,:] - targets[:,3:5,:,:]).pow(2), dim=1))

    loss = loss_obj*2 + loss_noobj*0.5 + loss_pos + loss_box

    return loss