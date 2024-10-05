
import os
import argparse
from models import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import pprint
from utils import *
from config import *
from tqdm import tqdm
from datasets.data_loaders import data_loader

def train():
    global args
    logger.info('\n' + pprint.pformat(args))
    if args.seed:
        seed_everything(args.seed + i)
        logger.info('seed: ', args.seed + i)

    if args.dataset == 'cifar10':
        epochs = 120
        lr = 0.01
        batch_size = 128
        weight_decay = 1e-4
        model = CNN(type='CIFAR10').to(device)
    elif args.dataset == 'cifar100':
        epochs = 200
        lr = 0.1
        batch_size = 128
        weight_decay = 1e-5
        model = ResNet34(num_classes=100).to(device)
    else:
        raise NotImplementedError

    train_loader, test_loader = data_loader(args=args, train_batch_size=batch_size, test_batch_size=batch_size*2, train_persistent=True, test_persistent=True)
    criterion = get_loss_config(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    for epoch in tqdm(range(epochs), ncols=60, desc=args.loss + ' ' + args.dataset):
        model.train()
        total_loss = 0.
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % args.test_freq == 0:
            test_acc1, _ = evaluate(test_loader, model, device)
            logger.info('Iter {}: loss={:.4f}, test_acc={:.4f}'.format(epoch+1, total_loss, test_acc1))

    if args.save_model:
        torch.save(model, results_path + '/model.pth')

    return test_acc1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eps-Softmax: Approximating One-Hot Vectors for Mitigating Label Noise')
    # dataset settings
    parser.add_argument('--dataset', type=str, default="cifar10", metavar='DATA', choices=['mnist', 'cifar10', 'cifar100'], help='dataset name')
    parser.add_argument('--root', type=str, default="../database/", help='the data root')
    parser.add_argument('--noise_type', type=str, default='symmetric', choices=['symmetric', 'asymmetric', 'dependent'], help='label noise type. using clean label by symmetric noise_rate=0')
    parser.add_argument('--noise_rate', type=str, default='0.8', help='the noise rate 0~1')
    # initialization settings
    parser.add_argument('--gpus', type=str, default='0', help='the used gpu id')
    parser.add_argument('--seed', type=int, default=None, help='initial seed')
    parser.add_argument('--trial_num', type=int, default=1, help='number of trials')
    parser.add_argument('--test_freq', type=int, default=1, help='epoch frequency to evaluate the test set')
    parser.add_argument('--save_model', default=False, action="store_true", help='whether to save trained model')
    parser.add_argument('--loss', type=str, default='ECEandMAE', help='the loss function: CE, ECEandMAE, EFLandMAE ... ')
    args = parser.parse_args()
    args.dataset = args.dataset.lower()
    if args.noise_rate == '0.0': args.noise_rate == '0'
    if args.noise_type == 'dependent':
        args.noise_method = '-' # noise_method is not about dependent noise
    if args.dataset == 'cifar10': # change root by yourself
        args.root = args.root + '/CIFAR10'
    elif args.dataset == 'cifar100':
        args.root = args.root + '/CIFAR100'
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if torch.cuda.is_available():  
        device = 'cuda' 
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
    print('We are using', device)

    results_path = os.path.join('./results/', args.dataset, args.loss, args.noise_type + '_' + args.noise_rate)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    logger = get_logger(results_path + '/log.txt')

    accs = []
    for i in range(args.trial_num):    
        acc = train()
        accs.append(acc)
    accs = torch.asarray(accs)*100
    logger.info(args.dataset+' '+args.loss+': %.2f±%.2f \n' % (accs.mean(), accs.std()))


    