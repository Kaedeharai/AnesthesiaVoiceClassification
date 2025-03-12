import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import *

def load_data_from_folder(folder_path):
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy'):
                file_paths.append(os.path.join(root, file))
    return file_paths

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.485, 0.485], [0.225, 0.225, 0.225])])

    folder_path = "D:\\AnesthesiaVoiceClassification\\data\\test\\mfcc"
    file_paths = load_data_from_folder(folder_path)
    
    # read class_indict
    json_path = "D:\\AnesthesiaVoiceClassification\\data\\test\\mfcc.json"
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # Extract unique labels and create a class index dictionary
    labels = sorted(set(item['label'] for item in data_list if 'label' in item))
    class_indict = {label: i for i, label in enumerate(labels)}
    reverse_class_indict = {i: label for label, i in class_indict.items()}

    # model = resnet50(num_classes=len(class_indict)).to(device)
    model = resnet18(num_classes=len(class_indict)).to(device)

    # load model weights
    weights_path = "resNet.pth"
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    
    # Load the pre-trained model weights
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)

    # Create a mapping from file names to labels
    file_to_label = {os.path.join(folder_path, item["featrue"]): item['label'] for i, item in enumerate(data_list)}
    # file_to_label = class_indict[file_to_label]


    # prediction
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for path in file_paths:
            data = np.load(path)
            if data.ndim == 2:
                data = np.stack([data] * 3, axis=-1)
            data = (data * 255).astype(np.uint8)
            img = Image.fromarray(data)
            
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            predict_cla = int(predict_cla)
            predict_label = reverse_class_indict[predict_cla]

            # Get the true label from the JSON data
            # file_name = os.path.basename(path)
            true_label = file_to_label[path]
            # true_label = class_indict[true_label]

            if predict_label == true_label:
                correct += 1
            total += 1

    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')
    # 0.3788

if __name__ == '__main__':
    main()