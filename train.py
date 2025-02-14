import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import *
from DataSet import *


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("using {} device.".format(device))

    batch_size = 1
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # print('Using {} dataloader workers every process'.format(nw))
    nw = 0

    mean = 0.0
    std = 0.225

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                                     transforms.Normalize([mean, mean, mean], [std, std, std])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
                                   transforms.Normalize([mean, mean, mean], [std, std, std])])}

    # method = "mel"
    method = "mfcc"

    train_dataset = AudioDataSet(data_path=os.path.join("data/train", method),
                                 json_path=os.path.join("data/train", method + ".json"),
                                 transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True,
                              num_workers=nw)


    validate_dataset = AudioDataSet(data_path=os.path.join("data/val", method),
                                    json_path=os.path.join("data/val", method + ".json"),
                                    transform=data_transform["train"])
    val_num = len(validate_dataset)
    validate_loader = DataLoader(validate_dataset,
                                 batch_size=batch_size, shuffle=False,
                                 num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))
    

    net = resnet50()
    model_weight_path = "resnet50pth.pth"
    net.load_state_dict(torch.load(model_weight_path))

    # in_channel = net.fc.in_features
    in_channel = 2048
    net.fc = nn.Linear(in_channel, 3)
    net.to(device)

    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.L1Loss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 3
    best_acc = 0.0
    save_path = './resNet.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()# retain_graph=True)
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)


            

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    
    
    # 释放未使用的 GPU 内存
    # torch.cuda.empty_cache()


if __name__ == '__main__':
    main()