
import os
import argparse
from models import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import pprint
from utils import *
from config import *
from tqdm import tqdm
from datasets.data_loaders import data_loader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Robust Loss Functions for Learning with Noisy Labels: Benchmark Datasets')
# dataset settings
parser.add_argument('--dataset', type=str, default="cifar100", choices=['mnist', 'cifar10', 'cifar100'], help='dataset name')
parser.add_argument('--root', type=str, default="../data", help='the dataset root, change root yourself')
parser.add_argument('--noise_type', type=str, default='symmetric', choices=['symmetric', 'asymmetric', 'dependent', 'human'], 
                    help='label noise type. human is the cifar-n dataset. using clean label by setting noise rate = 0')
parser.add_argument('--noise_rate', type=str, default='0.8', 
                    help='the noise rate 0~1. if using human noise, should set in [clean, worst, aggre, rand1, rand2, rand3, clean100, noisy100]')
parser.add_argument('--noise_method', type=str, default='method1', choices=['method1, method2'], 
                    help='different code implementation for symmetric and asymmetric noise, will cause little performance differences'
                         'this does not affect dependent and human noise')
# initialization settings
parser.add_argument('--gpus', type=str, default='0', help='the used gpu id')
parser.add_argument('--seed', type=int, default=123, help='initial seed')
parser.add_argument('--trials', type=int, default=1, help='number of trials')
parser.add_argument('--test_freq', type=int, default=1, help='epoch frequency to evaluate the test set')
parser.add_argument('--save_model', default=False, action="store_true", help='whether to save trained model')
# training settings
# loss: ECEandMAE: Eps-Softmax with CE loss (ECE) and MAE; EFLandMAE: Eps-Softmax with FL loss (EFL) and MAE
parser.add_argument('--loss', type=str, default='ECEandMAE', help='the loss function: CE, ECEandMAE, EFLandMAE, GCE ... ')
args = parser.parse_args()
args.dataset = args.dataset.lower()

# change root yourself
if args.dataset == 'cifar10': 
    args.root = args.root + '/cifar10'
elif args.dataset == 'cifar100':
    args.root = args.root + '/cifar100'

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
device = 'cuda'
torch.backends.cudnn.benchmark = True

def train(args, i):
    seed_everything(args.seed + i)
    # this codebase supports the simple mnist dataset for symmetric and asymmetric noise, but we do not use it in the paper
    if args.dataset == 'mnist': 
        epochs = 50
        lr = 0.01
        batch_size = 128
        model = CNN(type='mnist').to(device)
    elif args.dataset == 'cifar10':
        epochs = 120
        lr = 0.01
        batch_size = 128
        model = CNN(type='cifar10').to(device)
    elif args.dataset == 'cifar100':
        epochs = 200
        lr = 0.1
        batch_size = 128
        model = ResNet34(num_classes=100).to(device)
    else:
        raise NotImplementedError

    logger.info('\n' + pprint.pformat(args))
    l1_weight_decay, l2_weight_decay = get_weight_decay_config(args)
    logger.info('lr={}, batch_size={}, l1_weight_decay={}, l2_weight_decay={}'.format(lr, batch_size, l1_weight_decay, l2_weight_decay))
    
    train_loader, test_loader = data_loader(args=args, batch_size=batch_size)

    criterion = get_loss_config(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=l2_weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    for epoch in tqdm(range(epochs), ncols=60, desc=args.loss + ' ' + args.dataset):
        model.train()
        total_loss = 0.
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            if l1_weight_decay != 0:
                l1_decay = sum(p.abs().sum() for p in model.parameters())
                loss += l1_weight_decay * l1_decay
            loss.backward()
            # gradient norm bound, following previous work setting
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.) 
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % args.test_freq == 0:
            test_acc, _ = evaluate(test_loader, model, device)
            logger.info('Iter {}: loss={:.4f}, test_acc={:.4f}'.format(epoch, total_loss, test_acc))
            summary_writer.add_scalar('test_acc', test_acc, epoch+1)
    if args.save_model:
        torch.save(model, results_path + '/model.pth')
    # return last epoch test acc
    return test_acc 
    
if __name__ == "__main__":
    tag = f"default"
    results_path = os.path.join('./results', args.dataset, args.loss, args.noise_type + '_' + args.noise_rate, tag)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    logger = get_logger(results_path + '/result.log')
    summary_writer = SummaryWriter(log_dir=results_path)
    accs = []
    for i in range(args.trials):    
        acc = train(args, i)
        accs.append(acc)
    accs = torch.asarray(accs)*100
    logger.info(args.dataset+' '+args.loss+': %.2f±%.2f \n' % (accs.mean(), accs.std()))


    