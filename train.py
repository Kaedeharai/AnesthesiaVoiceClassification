import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Logger import *
from Model import *
from DataSet import *
from typing import Union
from torch.utils.tensorboard import SummaryWriter


def train_epoch(
        epoch: int,
        model: nn.Module, 
        device: Union[str, torch.device],
        loss_function: callable,
        optimizer: optim.Optimizer,
        training_loader: DataLoader,
        batch_nums_per_epoch: int,
        write: torch.utils.tensorboard.writer.SummaryWriter = None
        )-> float: 
    """
    Args:
        epoch (int): Current epoch number.
        model (nn.Module): PyTorch model to train.
        device (str | torch.device): Device to train on ("cuda" or "cpu").
        loss_function (callable): Loss function to use.
        optimizer (optim.Optimizer): Optimizer to use.
        train_loader (DataLoader): DataLoader for training data.
        batch_nums_per_epoch (int): Number of batches in per epoch of training set.
        write (torch.utils.tensorboard.writer.SummaryWriter): TensorBoard writer for logging.
    Returns:
        loss (float): Average training loss for the epoch.
    """

    model.train()
    running_loss = 0.0
    for step, train_data in enumerate(training_loader, start=0):
        data, labels = train_data
        optimizer.zero_grad()
        logits = model(data.to(device))

        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if write is not None:
            global_update_step = epoch * batch_nums_per_epoch + step
            write.add_scalar('per_batch_sum_train_loss', loss.item(), global_update_step)

    train_loss = running_loss / batch_nums_per_epoch
    if write is not None:
        write.add_scalar('per_epoch_average_train_loss', train_loss, epoch)
        write.close()

    return train_loss


def validate_epoch(
        epoch: int,
        model: nn.Module, 
        device: Union[str, torch.device],
        loss_function: callable,
        validation_loader: DataLoader,
        batch_nums_per_epoch: int,
        write: torch.utils.tensorboard.writer.SummaryWriter,
        ConfusionMatrixPath: str
        )-> float: 
    """
    Args:
        epoch (int): Current epoch number.
        model (nn.Module): PyTorch model to validate.
        device (str | torch.device): Device to validate on ("cuda" or "cpu").
        loss_function (callable): Loss function to use.
        validate_loader (DataLoader): DataLoader for validation data.
        batch_nums_per_epoch (int): Number of batches in per epoch of validation set.
        write (torch.utils.tensorboard.writer.SummaryWriter): TensorBoard writer for logging.
        ConfusionMatrixPath (str): Path to save confusion matrix plots.
    Returns:
        validation_accuracy (float): Average validation accuracy for the epoch.
    """

    model.eval()
    runing_accuracy = 0.0
    runing_loss = 0.0
    per_epoch_predict_label = []
    per_epoch_turth_label = []

    with torch.no_grad():
        for step, val_data in enumerate(validation_loader, start=0):
            data, ground_truth = val_data
            logits = model(data.to(device))

            predict_label = torch.argmax(logits, dim=1)
            batch_equal_nums = torch.eq(predict_label, ground_truth.to(device)).sum().item()
            runing_accuracy += batch_equal_nums

            loss = loss_function(logits, ground_truth.to(device))
            runing_loss += loss.item()

            per_epoch_predict_label.extend(predict_label.cpu().numpy())
            per_epoch_turth_label.extend(ground_truth.cpu().numpy())

            if write is not None:
                global_validation_step = epoch * batch_nums_per_epoch + step
                write.add_scalar('per_batch_sum_val_accuracy', batch_equal_nums, global_validation_step)
                write.add_scalar('per_batch_sum_val_loss', loss.item(), global_validation_step)

    validation_accuracy = runing_accuracy / batch_nums_per_epoch
    validation_loss = runing_loss / batch_nums_per_epoch
    if write is not None:
        write.add_scalar('per_epoch_average_val_accuracy', validation_accuracy, epoch)
        write.add_scalar('per_epoch_average_val_loss', validation_loss, epoch)
        write.close()

    if epoch % 10 == 0 and ConfusionMatrixPath is not None:
        cm = ConfusionMatrixCurve(per_epoch_turth_label, per_epoch_predict_label, num_classes=2)
        cm.plot(epoch, ConfusionMatrixPath)

    return validation_accuracy, validation_loss


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net, net_name = resnet50(num_classes=2)
    model_weight_path = "pth_files/resnet50pth.pth"
    state_dict = torch.load(model_weight_path, map_location=torch.device('cpu'))
    state_dict.pop('fc.weight', None)
    state_dict.pop('fc.bias', None)
    net.load_state_dict(state_dict, strict=False)
    net.to(device)

    method = "mel"
    # method = "mfcc"
    binary = True
    epochs = 200
    batch_size = 64
    lr=0.00001
    nw = min(os.cpu_count(), 8)
    best_accuracy = 0.0

    if binary:
        class_num = "bin"
    else:
        class_num = "multi"   
    pth_save_path = os.path.join('pth_files', f'{net_name}_{class_num}_{method}.pth')
    sys.stdout = Logger(f"Train_{net_name}_{class_num}_{method}", sys.stdout)
    write = SummaryWriter(os.path.join('runs', f'{net_name}_{class_num}_{method}'))
    ConfusionMatrixPath=os.path.join('Curve', f'{net_name}_{class_num}_{method}_ConfusionMatrix')
    os.makedirs(ConfusionMatrixPath, exist_ok=True)

    loss_function = nn.CrossEntropyLoss()
    params = net.parameters()
    optimizer = optim.Adam(params, lr, weight_decay=1e-4)


    train_dataset = AudioDataSet(data_path=os.path.join(method, "train"),
                                 json_path=os.path.join(method, "train.json"),
                                 binary=binary)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=nw,
                              shuffle=True,
                              pin_memory=True)
    train_steps = len(train_loader)

    validate_dataset = AudioDataSet(data_path=os.path.join(method, "val"),
                                    json_path=os.path.join(method, "val.json"),
                                    binary=binary)
    validate_loader = DataLoader(validate_dataset,
                                 batch_size=batch_size,
                                 num_workers=nw,
                                 shuffle=False,
                                 pin_memory=True)
    val_steps = len(validate_loader)


    for epoch in range(epochs):

        per_epoch_average_train_loss = train_epoch(epoch, 
                                                   net, 
                                                   device, 
                                                   loss_function, 
                                                   optimizer, 
                                                   train_loader, 
                                                   train_steps, 
                                                   write)
        print('[epoch %d] train_loss: %.3f' %(epoch + 1, per_epoch_average_train_loss))

        validation_accuracy, validation_loss = validate_epoch(epoch, 
                                                              net, 
                                                              device, 
                                                              loss_function, 
                                                              validate_loader, 
                                                              val_steps, 
                                                              write,
                                                              ConfusionMatrixPath)
        print('[epoch %d] val_loss: %.3f val_accuracy: %.3f' %(epoch + 1, validation_loss, validation_accuracy))

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            print('update best accuracy: %.3f' % best_accuracy)
            torch.save(net.state_dict(), pth_save_path)



if __name__ == '__main__':
    main()