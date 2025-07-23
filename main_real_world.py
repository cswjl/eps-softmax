
import os
import argparse
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from utils import *
from config import *
from lightning import Fabric
from datasets.data_loaders import webvision_loader, clothing1m_loader
import pprint
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Robust Loss Functions for Learning with Noisy Labels: Real-World Datasets')
# dataset settings
parser.add_argument('--dataset', type=str, default="webvision", choices=['webvision', 'clothing1m'], help='dataset name')
parser.add_argument('--root', type=str, default="../data", help='the data root')
# initialization settings
parser.add_argument('--gpus', type=str, default='0, 1, 2, 3', help='gup id, can multiple-gpu, change yourself')
parser.add_argument('--grad_bound', type=bool, default=True, help='the gradient norm bound, following previous work')
# training settings
parser.add_argument('--loss', type=str, default='ECEandMAE', help='the loss function: CE, ECEandMAE, EFLandMAE ... ')
args = parser.parse_args()
args.dataset = args.dataset.lower()

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
gpu_nums = torch.cuda.device_count() 
# using bf16-mixed precision for training
fabric = Fabric(accelerator='cuda', devices='auto', strategy='ddp', precision='bf16-mixed')
fabric.launch()

if args.dataset == 'webvision':
    lr = 0.4
    epochs = 250
    l1_weight_decay, l2_weight_decay = get_weight_decay_config(args)
    batch_size = int(256 / gpu_nums)
    # change root yourself in txt file in ./datasets
    args.grad_bound = True
    nesterov = True
    model = resnet50(num_classes=50, zero_init_residual=True)
    train_loader, test_loader, img_test_loader = webvision_loader(args, batch_size)

elif args.dataset == 'clothing1m':
    lr = 5e-3
    epochs = 10
    l1_weight_decay, l2_weight_decay = get_weight_decay_config(args)
    batch_size = int(256 / gpu_nums)
    # change root yourself
    args.root = args.root + '/clothing1m'
    args.grad_bound = False
    nesterov = False
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, 14)
    train_loader, test_loader = clothing1m_loader(args, batch_size)


if torch.distributed.get_rank() == 0:
    tag = f"default"
    results_path = os.path.join('./results', args.dataset, args.loss, tag)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    logger = get_logger(results_path + '/result.log')
    summary_writer = SummaryWriter(log_dir=results_path)

    logger.info('\n' + pprint.pformat(args))
    logger.info('lr={}, batch_size={}, l1_weight_decay={}, l2_weight_decay={}'.format(lr, batch_size, l1_weight_decay, l2_weight_decay))

criterion = get_loss_config(args)

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=nesterov, weight_decay=l2_weight_decay)
model, optimizer = fabric.setup(model, optimizer)
if args.dataset == 'webvision':
    scheduler = StepLR(optimizer, step_size=1, gamma=0.97)
    train_loader, test_loader, img_test_loader= fabric.setup_dataloaders(train_loader, test_loader, img_test_loader)
elif args.dataset == 'clothing1m':
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

if torch.distributed.get_rank() == 0:
    epochs_iterator = tqdm(range(epochs), ncols=60, desc=args.loss + ' ' + args.dataset)
else:
    epochs_iterator = range(epochs)

for epoch in epochs_iterator:
    model.train()
    total_loss = 0.
    for batch_x, batch_y in train_loader:
        model.zero_grad()
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        if l1_weight_decay != 0:
            l1_decay = sum(p.abs().sum() for p in model.parameters())
            loss += l1_weight_decay * l1_decay
        fabric.backward(loss)
        if args.grad_bound:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    test_acc1, test_acc5 = evaluate(test_loader, model)
    test_acc1, test_acc5 = fabric.all_gather(test_acc1).mean(), fabric.all_gather(test_acc5).mean()
    if args.dataset == 'webvision':
        img_acc1, img_acc5 = evaluate(img_test_loader, model)
        img_acc1, img_acc5 = fabric.all_gather(img_acc1).mean(), fabric.all_gather(img_acc5).mean()

    if torch.distributed.get_rank() == 0:
        if args.dataset == 'webvision':
            logger.info('Iter {}: loss={:.2f}, web_acc1={:.4f}, web_acc5={:.4f}, img_acc1={:.4f}, img_acc5={:.4f}'.format(epoch, total_loss, test_acc1, test_acc5, img_acc1, img_acc5))
            summary_writer.add_scalar('web_acc1', test_acc1, epoch+1)
            summary_writer.add_scalar('img_acc1', img_acc1, epoch+1)
        else:
            logger.info('Iter {}: loss={:.2f}, test_acc1={:.4f}, test_acc5={:.4f}'.format(epoch, total_loss, test_acc1, test_acc5))
            summary_writer.add_scalar('test_acc1', test_acc1, epoch+1)

