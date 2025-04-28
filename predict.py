import os
import json
import numpy as np
import torch

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

    folder_path = "mfccTest"
    file_paths = load_data_from_folder(folder_path)
    
    json_path = "mfccTest/mfccTest.json"
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    labels = sorted(set(item['label'] for item in data_list if 'label' in item))
    class_indict = {label: i for i, label in enumerate(labels)}
    reverse_class_indict = {i: label for label, i in class_indict.items()}


    # model = resnet50(num_classes=len(class_indict)).to(device)
    model = resnet18(num_classes=2).to(device)
    weights_path = "resNet18_bin.pth"
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)

    file_to_label = {os.path.join(folder_path, item["featrue"]): item['label'] for i, item in enumerate(data_list)}
    # file_to_label = class_indict[file_to_label]


    # prediction
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for path in file_paths:
            data = np.load(path)
            
            data = np.stack([data] * 3, axis=-1)
            data = data.astype(np.uint8)

            data = torch.tensor(data, dtype=torch.float32)
            data = data.permute(2, 0, 1)
            
            # img = data_transform(img)
            # expand batch dimension
            data = torch.unsqueeze(data, dim=0)

            # predict class
            output = torch.squeeze(model(data.to(device))).cpu()
            predict = torch.softmax(output, dim=1)
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
    # 0.4024
    # 0.4296

if __name__ == '__main__':
    main()