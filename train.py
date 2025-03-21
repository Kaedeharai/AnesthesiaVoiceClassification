import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import *
from DataSet import *

from torch.utils.tensorboard import SummaryWriter
write = SummaryWriter('runs/resnet18_binary')


class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        # print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag
        # self.log = open(filename, 'a+')

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("using {} device.".format(device))

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 6])
    # print('Using {} dataloader workers every process'.format(nw))
    # nw = 0

    sys.stdout = Logger("TrainBin.log", sys.stdout)


    # mean = 0.485
    # std = 0.225

    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(224),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    #                                 # transforms.Normalize([mean, mean, mean], [std, std, std])]),
    #     "val": transforms.Compose([transforms.Resize(256),
    #                                transforms.CenterCrop(224),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    #                                # transforms.Normalize([mean, mean, mean], [std, std, std])])}

    # method = "mel"
    method = "mfcc"

    train_dataset = AudioDataSet(data_path=os.path.join(method, "train"),
                                 json_path=os.path.join(method, "train.json"))
                                #  transform=data_transform["train"])

    train_num = len(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=False,
                              num_workers=nw)


    validate_dataset = AudioDataSet(data_path=os.path.join(method, "val"),
                                    json_path=os.path.join(method, "val.json"))
                                    # transform=data_transform["train"])

    val_num = len(validate_dataset)
    validate_loader = DataLoader(validate_dataset,
                                 batch_size=batch_size, shuffle=False,
                                 num_workers=nw)


    net = resnet18(num_classes=2)
    model_weight_path = "resNet18_bin.pth"

    state_dict = torch.load(model_weight_path, map_location=torch.device('cpu'))
    state_dict.pop('fc.weight', None)
    state_dict.pop('fc.bias', None)
    net.load_state_dict(state_dict, strict=False)
    net.to(device)


    loss_function = nn.CrossEntropyLoss()

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001)

    epochs = 200
    best_acc = 0.0
    save_path = './resNet18_bin.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        # train_bar = tqdm(train_loader, file=sys.stdout)


        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            write.add_scalar('train_loss', loss.item(), epoch * train_steps + step)

            print('[epoch %d, batch %d] loss: %.03f' %
                  (epoch + 1, step + 1, loss.item()))
            # train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
            #                                                          epochs,
            #                                                          loss)

            if step % 10 == 0:
                write.add_scalar('train_loss_curve', running_loss / (step + 1), epoch * train_steps + step)


        net.eval()
        # accumulate accurate number / epoch
        acc = 0.0
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)

                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                write.add_scalar('acc', acc, epoch * train_steps + step)
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()
                val_bar.desc = "valid epoch[{}/{}] loss:{:.3f} acc:{:.3f}".format(epoch + 1,
                                                                                  epochs,
                                                                                  loss,
                                                                                  acc)

                write.add_scalar('val_loss_curve', val_loss / len(validate_loader), epoch)



        val_accurate = acc / val_num
        write.add_scalar('val_accuracy', val_accurate, epoch)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    write.close()
    
    
    # 释放未使用的 GPU 内存
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()



#TODO draw train loss curve, val loss curve,
#TODO every 10 epochs, val acc --> curve, confusion matrix sklearn --> figure, json, classification_report


#TODO train dataset shuffle, augmentation --> train model, val --> train dataset, val loss decrease very low
#TODO dataset --> normalization , min_max . augmenatation


#TODO model ? 