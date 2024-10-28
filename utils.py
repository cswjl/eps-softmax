import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import logging
from losses import *
from torch.autograd import Variable
from torchvision.transforms import RandAugment
from torch.utils.data import Dataset 
import copy
import PIL.Image as Image

@torch.no_grad()
def evaluate(loader, model, device=None):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    for inputs, targets in loader:
        if device:
            inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)

        _, top5_preds = probabilities.topk(5, dim=1)
        top1_preds = top5_preds[:, :1]  

        correct_top1 += top1_preds.eq(targets.view(-1, 1)).sum().item()
        correct_top5 += top5_preds.eq(targets.view(-1, 1).expand_as(top5_preds)).sum().item()

        total += targets.size(0)

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    return top1_acc, top5_acc

def get_logger(filename):
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=filename, format=head, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger


def log(f=None, p='\n', is_print=True):
    if f is None:
        if is_print:
            print(p)
    else:
        if is_print:
            print(p)
        f.write(p + '\n')


def save_accs(path, label, accs):
    with open(os.path.join(path, label+'.csv'), 'w') as f:
        m = accs.shape[0]
        f.write(','.join(['test ' + str(i+1) for i in range(m)]) + '\n')
        for i in range(accs.shape[1]):
            f.write(','.join([str(f) for f in accs[:,i]]) + '\n')

def save_acc(path, label, accs):
    with open(os.path.join(path, label+'.csv'), 'w') as f:
        for a in accs:
            f.write(str(a) + '\n')

def adjust_learning_rate(optimizer, epoch, args): #not use this
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * (0.97 ** (epoch - args.warmup_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def seed_everything(seed: int):
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed (for both CPU and GPU)
    torch.manual_seed(seed)
    
    # If you are using a GPU, also set the seed for all GPUs (if you have multiple GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    
    # Ensure deterministic behavior for some PyTorch operations
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # For reproducibility in DataLoader shuffling (if you are using DataLoader)
    # Ensure workers in DataLoader are seeded correctly
    def worker_init_fn(worker_id):
        # Each worker will have its own distinct seed based on the global seed and worker_id
        np.random.seed(seed + worker_id)

    return worker_init_fn

def predict_softmax(predict_loader, model):

    model.eval()
    softmax_outs = []
    loss_outs = []
    with torch.no_grad():
        for images1, images2, images3, target in predict_loader:
            images1 = Variable(images1).cuda()
            images2 = Variable(images2).cuda()
            target = target.cuda()
            logits1 = model(images1)
            logits2 = model(images2)
            outputs = (F.softmax(logits1, dim=1) + F.softmax(logits2, dim=1)) / 2
            softmax_outs.append(outputs)
            loss = F.cross_entropy(logits1, target, reduction='none') + F.cross_entropy(logits2, target, reduction='none')
            loss = loss / 2
            loss_outs.append(loss)
        
    return torch.cat(softmax_outs, dim=0).cpu(), torch.cat(loss_outs, dim=0).cpu()

class Semi_Labeled_Dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = np.array(data)
        self.targets = np.array(targets)
        self.length = len(self.targets)
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.transform2 = copy.deepcopy(self.transform)
        self.transform2.transforms.insert(0, RandAugment(3,5))
        

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            weak_out1 = self.transform(img)
            weak_out2 = self.transform(img)
            strong_out = self.transform2(img)
        return weak_out1, weak_out2, strong_out, target

    def __len__(self):
        return self.length

    def getData(self):
        return self.data, self.targets


class Semi_Unlabeled_Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = np.array(data)
        self.length = self.data.shape[0]
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.transform2 = copy.deepcopy(self.transform)
        self.transform2.transforms.insert(0, RandAugment(3,5))

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            weak_out1 = self.transform(img)
            weak_out2 = self.transform(img)
            strong_out = self.transform2(img)
        return weak_out1, weak_out2, strong_out

    def __len__(self):
        return self.length

    def getData(self):
        return self.data

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)