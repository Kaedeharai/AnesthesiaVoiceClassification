import os
import json
from torch.utils.data import Dataset
import numpy as np
from Preprocessing import *
import torch


class AudioDataSet(Dataset):

    classes = ["before", "after", "recovery"]
    default_class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __init__(self, 
                 data_path:str, 
                 json_path:str, 
                 binary:bool=True,
                 transform=None, 
                 class_to_idx=None, 
                 method=None):

        self.transform = transform
        self.data_path = data_path
        self.class_to_idx = class_to_idx if class_to_idx is not None else self.default_class_to_idx

        with open(json_path, 'r', encoding="utf-8") as f:
            self.json_dict = json.load(f)

        if binary:
            self.json_bin = []
            for data in self.json_dict:
                if data['label'] == "before" or data['label'] == "after":
                    self.json_bin.append(data)
                    data['label'] = self.class_to_idx[data['label']]
            self.json_file = self.json_bin
        else:
            self.json_thr = []
            for data in self.json_dict:
                self.json_thr.append(data)
                data['label'] = self.class_to_idx[data['label']]
            self.json_file = self.json_thr


    def __len__(self):

        return len(self.json_file)


    def __getitem__(self, index): 

        data = self.json_bin[index]
        featrue_path = os.path.join(self.data_path, data['featrue'])
        Label = data['label']
        Label = torch.tensor(Label, dtype=torch.long)

        Data = np.load(featrue_path)
        Data = np.stack([Data] * 3, axis=-1)
        Data = Data.astype(np.uint8)
        Data = torch.tensor(Data, dtype=torch.float32)
        Data = Data.permute(2, 0, 1)

        # feature = torch.tensor(feature, dtype=torch.float32)
        # if self.transform is not None:
        #     feature = self.transform(feature)

        return Data, Label