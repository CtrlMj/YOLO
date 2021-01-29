import torch
import numpy as np
import torch.nn as nn
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import transforms
from YOLOmodel import YOLO, YOLOLoss
from YOLOData import Data
from utilsme import mean_average_precision
from utilsme import listOfboxes
from utilsme import transform
from tqdm.notebook import tqdm
device = torch.device("cuda")

def train(model, optimizer, loss_function, traindata_loader):
    model.train()
    total_loss, counter = 0, 0
    progressor = tqdm(traindata_loader)
    for train_x, targets in progressor:
        train_x, targets = train_x.to(device), targets.to(device)
        predictions = model(train_x)
        loss = loss_function(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        counter += 1
        progressor.set_postfix(train_loss= total_loss/counter)
    return total_loss / counter
    
    
def test(model, loss_function, testdata_loader):
    model.eval()
    total_loss, counter = 0, 0
    progressor = tqdm(testdata_loader)
    with torch.no_grad():    
        for test_x, targets in progressor:
            test_x, targets = test_x.to(device), targets.to(device)
            predictions = model(test_x)
            total_loss += loss_function(predictions, targets).item()
            counter += 1
            progressor.set_postfix(test_loss= total_loss/counter)
    return total_loss / counter
    
    

transformations = transform([transforms.Resize((448, 448)), transforms.ToTensor()])
traindata = Data(image_path='archive_2/images', label_path='archive_2/labels', annot_path='archive_2/8examples.csv', transform=transformations)
testdata = Data(image_path='archive_2/images', label_path='archive_2/labels', annot_path='archive_2/test.csv', transform=transformations)
traindata_loader = DataLoader(traindata, batch_size=2, shuffle=True)
testdata_loader = DataLoader(testdata, batch_size=16, shuffle=True)


def save_checkpoint(state_dict, is_best, path):
    checkpoint_path = path + '/checkpoint.pt'
    torch.save(state_dict, checkpoint_path)
    if is_best:
        best_path = path + "/best_state.pt"
        shutil.copyfile(checkpoint_path, best_path)
def get_lr(optimizer):
    for param_grp in optimizer.param_groups:
        return param_grp['lr']
writer = SummaryWriter("logs")




Epochs = 100
lr = 0.00002
weight_decay = 0
yolo = YOLO(in_channels=3).to(device)
yololoss = YOLOLoss().to(device)
optimizer = AdamW(yolo.parameters(), lr=lr)
best_MAP = 0
is_best = False
for epoch in range(Epochs):
    print(f"Epoch numba {epoch} ############################################### best MAP {best_MAP}")
    
    prediction_boxes, target_boxes = listOfboxes(yolo, traindata_loader, iou_threshold=0.5, obj_threshold=0.02, S=7)
    
    precision = mean_average_precision(prediction_boxes, target_boxes, iou_threshold=0.5, num_classes=20)
    writer.add_scalar("MAP", precision, epoch)
    if precision > best_MAP:
        best_MAP = precision
        is_best = True
    state_dict = {'model_state': yolo.state_dict(), 'optimizer_state': optimizer.state_dict(), 'epoch': epoch}
    save_checkpoint(state_dict, is_best, './Checkpoints')
    
    trainloss = train(yolo, optimizer, yololoss, traindata_loader)
    writer.add_scalar("loss/train", trainloss, epoch)
#     testloss = test(yolo, yololoss, testdata_loader)
#     writer.add_scalar("loss/eval", testloss, epoch)


import tensorboard
%reload_ext tensorboard
%tensorboard --logdir logs --port 6006
