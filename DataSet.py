import os
import json
from torch.utils.data import Dataset
import numpy as np
from Preprocessing import *
from PIL import Image
import torch


class AudioDataSet(Dataset):

    classes = ["before", "after", "recovery"]
    default_class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __init__(self, data_path, json_path, transform=None, class_to_idx=None):

        self.transform = transform
        self.data_path = data_path
        self.class_to_idx = class_to_idx if class_to_idx is not None else self.default_class_to_idx
        
        with open(json_path, 'r', encoding="utf-8") as f:
            self.json_dict = json.load(f)


        for data in self.json_dict:
            if data['label'] not in self.class_to_idx:
                raise ValueError(f"Label '{data['label']}' not found in class_to_idx mapping.")
            data['label'] = self.class_to_idx[data['label']]

    def __len__(self):
        length = 0
        for data in self.json_dict:
            if data["label"] != 2:
                length += 1
        return length
    
    def __getitem__(self, index):
        while True:
            data = self.json_dict[index]
            file_path = os.path.join(self.data_path, data["featrue"])

            Data = np.load(file_path)

            if data["label"] != 2:
                if Data.ndim == 2:
                    Data = np.stack([Data] * 3, axis=-1)
                Data = (Data * 255).astype(np.uint8)
                Data = Image.fromarray(Data)

                label = data["label"]
                label = torch.tensor(label, dtype=torch.long)

                if self.transform:
                    Data = self.transform(Data)

                return Data, label

            else:
                index = (index + 1) % len(self.json_dict)
