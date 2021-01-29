import pandas as pd
from PIL import Image
import torch
import os

class Data(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, annot_path, transform, split=7, n_BBs=2, n_Class=20):
        super(Data, self).__init__()
        self.split = split
        self.n_BBs = n_BBs
        self.n_class = n_Class
        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform
        self.annots = pd.read_csv(annot_path)
        
    def __len__(self):
        return len(self.annots)
    
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_path, self.annots.iloc[index, 0]))
        image = self.transform(image)
        labelpath = os.path.join(self.label_path, self.annots.iloc[index, 1])
        objects = []
        with open(labelpath) as f:
            for line in f.readlines():
                obj = [float(x) if for x in line.strip().split()]
                objects.append(obj)
        
        label = torch.zeros((self.split, self.split, self.n_class + 5*self.n_BBs))
        for obj in objects:
            clas, relativex, relativey, width, height = obj[0], obj[1], obj[2], obj[3], obj[4]
            i = int(relativey * self.split)
            j = int(relativex * self.split)
            x = relativex*self.split - j
            y = relativey*self.split - i
            cellwidth = self.split * width
            cellheight = self.split * height
            
            if label[i, j, self.n_class] == 0:
                label[i, j, self.n_class] = 1
                label[i, j, clas] = 1
                label[i, j, self.n_class + 1] = x
                label[i, j, self.n_class + 2] = y
                label[i, j, self.n_class + 3] = cellwidth
                label[i, j, self.n_class + 4] = cellheight
            
        
        return image, label
