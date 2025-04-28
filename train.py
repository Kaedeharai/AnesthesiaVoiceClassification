import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from logger import *
from model import *
from DataSet import *
from torch.utils.tensorboard import SummaryWriter
write = SummaryWriter('runs/resnet18_binary_mfcc_sec')


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 6])
    sys.stdout = Logger("Train18Bin_mfcc_sec.log", sys.stdout)

    method = "mel"
    # method = "mfcc"

    train_dataset = AudioDataSet(data_path=os.path.join(method, "train"),
                                 json_path=os.path.join(method, "train.json"))
    train_num = len(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=nw)

    validate_dataset = AudioDataSet(data_path=os.path.join(method, "val"),
                                    json_path=os.path.join(method, "val.json"))
    val_num = len(validate_dataset)
    validate_loader = DataLoader(validate_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=nw)


    net = resnet18(num_classes=2)
    model_weight_path = "./resNet18_bin_mfcc.pth"

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
    save_path = './resNet18_bin_mfcc_sec.pth'
    train_steps = len(train_loader)
    
    for epoch in range(epochs):
        all_preds = []
        all_labels = []
        val_acc = 0.0
        val_loss = 0.0
        train_loss = 0.0


        net.train()
        for step, data in enumerate(train_loader, start=0):
            images, labels = data

            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            write.add_scalar('perbatch_train_loss', loss.item(), epoch * train_steps + step)
            write.add_scalar('average_train_loss_curve', train_loss / (step + 1), epoch * train_steps + step)
            
            print('[epoch %d, batch %d] batchloss: %.03f' %(epoch + 1, step + 1, loss.item()))


        net.eval()
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                logits = net(val_images.to(device))
                # probs = F.softmax(logits, dim=1)
                _, predicted = torch.max(logits, dim=1)

                perbatch_acc = torch.eq(predicted, val_labels.to(device)).sum().item()
                write.add_scalar('perbatch_val_acc', perbatch_acc, epoch * train_steps + step)

                val_acc += perbatch_acc
                write.add_scalar('sum_acc', val_acc, epoch * train_steps + step)

                loss = loss_function(logits, val_labels.to(device))
                val_loss += loss.item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
                val_bar.desc = "valid epoch[{}/{}] batchloss:{:.3f}".format(epoch + 1,
                                                                       epochs,
                                                                       loss.item())

                write.add_scalar('val_loss_curve', val_loss / len(validate_loader), epoch)

        if epoch % 10 == 0:
            cm = ConfusionMatrixCurve(all_labels, all_preds, num_classes=2)
            cm.plot(epoch, "18_binary_mfcc_sec")


        val_accurate = val_acc / val_num
        write.add_scalar('val_accuracy', val_accurate, epoch)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %(epoch + 1, train_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    write.close()


if __name__ == '__main__':
    main()