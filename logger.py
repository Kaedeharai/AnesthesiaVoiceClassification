from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os


class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
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


class ConfusionMatrixCurve():
    def __init__(self, all_labels, all_preds, num_classes=2):
        self.all_labels = all_labels
        self.all_preds = all_preds
        self.num_classes = num_classes

    def plot(self, epoch, save_path):
        os.makedirs(save_path, exist_ok=True)
        cm = confusion_matrix(self.all_labels, self.all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Class {i}' for i in range(self.num_classes)],
                    yticklabels=[f'Class {i}' for i in range(self.num_classes)])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(save_path, f'confusion_matrix_epoch_{epoch}.png'))
        # plt.show()
        plt.close()


class AccuracyCurve():
    def __init__(self, all_labels, all_preds):
        self.all_labels = all_labels
        self.all_preds = all_preds

    def plot(self, epoch, save_path):
        os.makedirs(save_path, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(self.all_labels, label='True Labels', color='blue')
        plt.plot(self.all_preds, label='Predicted Labels', color='orange')
        plt.xlabel('Sample Index')
        plt.ylabel('Label')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.savefig(os.path.join(save_path, f'accuracy_curve_{epoch}.png'))
        plt.show()
        plt.close()


class PrecisionRecallCurve():
    def __init__(self, all_labels, all_preds):
        self.all_labels = all_labels
        self.all_preds = all_preds

    def plot(self, epoch, save_path):
        os.makedirs(save_path, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(self.all_labels, label='True Labels', color='blue')
        plt.plot(self.all_preds, label='Predicted Labels', color='orange')
        plt.xlabel('Sample Index')
        plt.ylabel('Label')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(os.path.join(save_path, f'precision_recall_curve_{epoch}.png'))
        plt.show()
        plt.close()

