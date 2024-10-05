
import os
import torch
import torch.nn.functional as F
import logging
from losses import *
import random


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

        # 只计算一次Top-5，然后从中提取Top-1
        _, top5_preds = probabilities.topk(5, dim=1)
        top1_preds = top5_preds[:, :1]  # Top-1是Top-5中的第一个

        correct_top1 += top1_preds.eq(targets.view(-1, 1)).sum().item()
        correct_top5 += top5_preds.eq(targets.view(-1, 1).expand_as(top5_preds)).sum().item()

        total += targets.size(0)

    # 计算并返回Top-1和Top-5准确率
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