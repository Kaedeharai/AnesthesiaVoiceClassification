import os
import json
from torch.utils.data import Dataset
import numpy as np
from Preprocessing import *
from PIL import Image
import torch


class AudioDataSet(Dataset):

    # classes = ['0 - first', '1 - second', '2 - third']
    # class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __init__(self, data_path, json_path, transform=None):

        self.transform = transform
        self.data_path = data_path
        
        with open(json_path, 'r', encoding="utf-8") as f:
            self.json_dict = json.load(f)
        
        # for data in self.json_dict:
            # data['label'] = self.class_to_idx[data['label']]
        

    def __len__(self):
        return len(self.json_dict)
    
    def __getitem__(self, index):

        data = self.json_dict[index]
        file_path = data["featrue"]

        Data = np.load(file_path)
        if Data.ndim == 2:
            Data = np.stack([Data] * 3, axis=-1)  # 将单通道图像复制为伪 RGB 图像
        Data = (Data * 255).astype(np.uint8)
        Data = Image.fromarray(Data)
        label = int(data["label"])
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            Data = self.transform(Data)
        return Data, label


