import os
import argparse
import torch
import torch.nn.functional as F
from models import *
from losses import *
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR
import random
import pprint
from utils import *
from config import *
from tqdm import tqdm
from utils import Semi_Labeled_Dataset, Semi_Unlabeled_Dataset, AverageMeter, predict_softmax
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from datasets.data_method1.build_dataset import build_dataset_method1
from datasets.data_human.bulid_dataset import build_dataset_human

parser = argparse.ArgumentParser(description='Method with Semi-Supervised Learning')
# dataset settings
parser.add_argument('--dataset', type=str, default="cifar100", choices=['cifar10', 'cifar100'], help='dataset name')
parser.add_argument('--root', type=str, default="../data", help='the data root')
parser.add_argument('--noise_type', type=str, default='human', choices=['symmetric', 'asymmetric', 'human'], help='the noise type')
parser.add_argument('--noise_rate', type=str, default='noisy100', help='the noise rate'
'human: cifar10: clean, aggre, worst, rand1, rand2, rand3 | cifar100: clean100, noisy100')
# initialization settings
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--seed', type=int, default=123, help='initial seed')
parser.add_argument('--save', default='./results', type=str)
parser.add_argument('--trials', type=int, default=1)
# training settings
# ECEandMAE(semi): Eps-Softmax with CE loss (ECE) and MAE with Semi-supervised learning
parser.add_argument('--loss', type=str, default='ECEandMAE(Semi)', help='method name')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
device = 'cuda'
torch.backends.cudnn.benchmark = True


criterion = ECEandMAE(m=10000, alpha=0.5, beta=1)
criterion2 = ECEandMAE(m=10, alpha=1, beta=1)

weight_decay = 5e-4
lr = 0.1
epochs = 300
batch_size = 128
warm_epoch = 65
lamb_u = 1
threshold = 0.2

if args.dataset == 'cifar10':
    args.root = args.root + '/cifar10'
    num_classes = 10
    if args.noise_type == 'human': 
        if args.noise_rate == 'worst':
            k = 2500
        else:
            k = 3500
    else:
        if args.noise_rate == '0.2':
            k = 3500
        elif args.noise_rate == '0.4':
            k = 2500
        elif args.noise_rate == '0.5':
            k = 2000
        elif args.noise_rate == '0.8':
            k = 1000
        else:
            raise ValueError('No default parameter for this case, set yourself!')

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
   
elif args.dataset == 'cifar100':
    args.root = args.root + '/cifar100'
    num_classes = 100
    if args.noise_type == 'human': 
        if args.noise_rate == 'noisy100':
            k = 250
        else:
            k = 350
    else:
        if args.noise_rate == '0.2':
            k = 350
        elif args.noise_rate == '0.4':
            k = 250
        elif args.noise_rate == '0.5':
            k = 200
        elif args.noise_rate == '0.8':
            k = 100
        else:
            raise ValueError('No default parameter for this case, set yourself!')

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
else:
    raise ValueError('Invalid value {}'.format(args.dataset))

def mixup_loss(x, y, model, criterion, mix_weight=1.0):
    '''Compute the mixup data. Returns mixed inputs, pairs of targets, and lambda'''
    if mix_weight > 0:
        lam = np.random.beta(mix_weight, mix_weight)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    out = model(mixed_x)

    loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
    return loss

def evaluate(loader, model):
    model.eval()
    correct = 0.
    total = 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        z = model(x)
        probs = F.softmax(z, dim=1)
        pred = torch.argmax(probs, 1)
        total += y.size(0)
        correct += (pred==y).sum().item()

    acc = float(correct) / float(total)
    return acc

def evaluate2(loader, model1, model2):
    model1.eval()
    model2.eval()
    correct1 = 0.
    correct2 = 0.
    avg_correct = 0.
    total = 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        z1 = model1(x)
        probs1 = F.softmax(z1, dim=1)
        pred1 = torch.argmax(probs1, 1)
        total += y.size(0)
        correct1 += (pred1==y).sum().item()

        z2 = model2(x)
        probs2 = F.softmax(z2, dim=1)
        pred2 = torch.argmax(probs2, 1)
        correct2 += (pred2==y).sum().item()
        
        avg_probs = (probs1 + probs2) / 2
        avg_pred = torch.argmax(avg_probs, 1)
        avg_correct += (avg_pred==y).sum().item()

    acc1 = float(correct1) / float(total)
    acc2 = float(correct2) / float(total)
    avg_acc = float(avg_correct) / float(total)
    return acc1, acc2, avg_acc

def linear_rampup(current):
    current = np.clip(current / 200, 0.0, 1.0)
    return lamb_u * float(current)

# Semi Training
def Match_train(epoch, net, optimizer, labeled_trainloader, unlabeled_trainloader, criterion):
    if epoch < 100:
        mix_w = 0.75
    else:
        mix_w = 4
    net.train()
    losses = AverageMeter('Loss', ':6.2f')

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = int(50000 / (batch_size))
    for i in range(num_iter):
        try:
            inputs_x, inputs_x2, inputs_x3, targets_x = next(labeled_train_iter)
        except StopIteration:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, inputs_x2, inputs_x3, targets_x = next(labeled_train_iter)

        try:
            inputs_u, inputs_u2, inputs_u3 = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3 = next(unlabeled_train_iter)

        # batch_size = inputs_x.size(0)
        # targets_x = torch.zeros(batch_size, num_class).scatter_(1, targets_x.view(-1, 1), 1)
        inputs_x, inputs_x2, inputs_x3, targets_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), targets_x.cuda()
        inputs_u, inputs_u2, inputs_u3 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()
    
        with torch.no_grad():
            outputs_u = net(inputs_u)
            outputs_u2 = net(inputs_u2)
            probs_u = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            max_probs, targets_u = torch.max(probs_u, dim=1)
            targets_u = targets_u.detach()
            mask = max_probs.ge(threshold)
        batch_x = torch.cat([inputs_x, inputs_u3[mask]], dim=0)
        batch_y = torch.cat([targets_x, targets_u[mask]], dim=0)

        if i == 1:
            print('len_x:{}, len_u:{}, len_u_mask:{}'.format(len(inputs_x), len(inputs_u3), len(inputs_u3[mask])))

        idx = torch.randperm(batch_x.size(0))
        batch_x = batch_x[idx]
        batch_y = batch_y[idx]
        x_l = batch_x[:len(inputs_x)]
        y_l = batch_y[:len(inputs_x)]
        x_u = batch_x[len(inputs_x):]
        y_u = batch_y[len(inputs_x):]
        batch_x_l = torch.cat([inputs_x, x_l], dim=0)
        batch_y_l = torch.cat([targets_x, y_l], dim=0)
        batch_x_u = torch.cat([inputs_u3, x_u], dim=0)
        batch_y_u = torch.cat([targets_u, y_u], dim=0)

        Lcls = mixup_loss(batch_x_l, batch_y_l, net, criterion, mix_w)
        Lu = mixup_loss(batch_x_u, batch_y_u, net, criterion, mix_w)
        loss = Lcls + linear_rampup(epoch) * Lu
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), len(batch_x))

    print(losses)

def select_samples(losses, labels, k):
    # Step 1: For each class, select the k samples with the smallest loss.
    good_samples_indices = set()
    # print(labels)
    for class_idx in range(num_classes):
        # Find the indices of the samples belonging to the current class
        class_indices = np.where(labels == class_idx)[0]
        class_losses = losses[class_indices]
        # print(class_indices, losses)
        k_t = k
        if k_t > len(class_losses):
            k_t = len(class_losses) - 20
        # Sort by loss and get the indices of the k samples with the smallest loss
        # print(class_idx, class_losses.shape, k_t)
        _, class_good_indices = torch.topk(class_losses, k_t, dim=0, largest=False)
        good_samples_indices.update(class_indices[class_good_indices].tolist())

    all_indices = set(range(len(labels)))
    bad_samples_indices = list(all_indices - good_samples_indices)

    return list(good_samples_indices), bad_samples_indices

def update_trainloader(model, semi_train_loader):
    soft_outs, losses = predict_softmax(semi_train_loader, model)
    train_dataset = semi_train_loader.dataset
    train_data = train_dataset.data
    train_targets = train_dataset.targets

    confident_indexs, unconfident_indexs = select_samples(losses, train_targets, k)
    confident_dataset = Semi_Labeled_Dataset(train_data[confident_indexs], train_targets[confident_indexs], train_transform)
    unconfident_dataset = Semi_Unlabeled_Dataset(train_data[unconfident_indexs], train_transform)
    uncon_batch = int(batch_size / 2) if len(unconfident_indexs) > len(confident_indexs) else int(len(unconfident_indexs) / (len(confident_indexs) + len(unconfident_indexs)) * batch_size)
    con_batch = batch_size - uncon_batch
    labeled_trainloader = DataLoader(dataset=confident_dataset, batch_size=con_batch, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    unlabeled_trainloader = DataLoader(dataset=unconfident_dataset, batch_size=uncon_batch, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    return labeled_trainloader, unlabeled_trainloader

def run(args):
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    logger.info('batch_size={}, lr={:.2f}'.format(batch_size, lr))

    if args.noise_type == 'asymmetric' or args.noise_type == 'symmetric':
        noise_rate = float(args.noise_rate)
        train_dataset, test_dataset = build_dataset_method1(args.dataset, args.root, args.noise_type, noise_rate, train_transform, test_transform)
        train_data, train_targets = train_dataset.data, train_dataset.targets
    elif args.noise_type == 'human':
        noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
        noise_type = noise_type_map[args.noise_rate]
        # load dataset
        if args.dataset == 'cifar10':
            args.noise_path = './datasets/data_human/CIFAR-10_human.pt'
        elif args.dataset == 'cifar100':
            args.noise_path = './datasets/data_human/CIFAR-100_human.pt'
        else: 
            raise NameError(f'Undefined dataset {args.dataset}')
        train_dataset, test_dataset = build_dataset_human(args.dataset, args.root, noise_type, args.noise_path, train_transform, test_transform)
        train_data = train_dataset.train_data
        if noise_type == 'clean_label':
            train_targets = train_dataset.train_labels
        else:
            train_targets = train_dataset.train_noisy_labels

    semi_train_dataset = Semi_Labeled_Dataset(train_data, train_targets, train_transform)
    random_semi_train_loader = DataLoader(dataset=semi_train_dataset,
                                batch_size=batch_size,
                                num_workers=16,
                                shuffle=True,
                                pin_memory=True,
                                persistent_workers=True)
    
    semi_train_loader = DataLoader(dataset=semi_train_dataset,
                                batch_size=batch_size,
                                num_workers=16,
                                shuffle=False,
                                pin_memory=True,
                                persistent_workers=True)
        
    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size*2,
                                shuffle=False,
                                num_workers=16,
                                pin_memory=True,
                                persistent_workers=True)
    if args.noise_type == 'human':
        model1 = ResNet34(num_classes=num_classes).to(device)
        model2 = ResNet34(num_classes=num_classes).to(device)
    else:
        raise ValueError('No default parameter for this case, set yourself!')
        
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler1 = MultiStepLR(optimizer1, milestones=[60, 160, 260], gamma=0.1)
    scheduler2 = MultiStepLR(optimizer2, milestones=[60, 160, 260], gamma=0.1)
    
    best_acc = 0.
    for epoch in tqdm(range(epochs), ncols=60, desc=args.loss + ' ' + args.dataset):
        model1.train()
        model2.train()
        test_acc1 = 0.
        test_acc2 = 0.
        avg_acc = 0.
        if epoch < warm_epoch:
            for batch_x1, batch_x2, _, batch_y in random_semi_train_loader:
                batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)
                optimizer1.zero_grad()
                out1 = model1(batch_x1)
                loss1 = criterion(out1, batch_y)
                loss1.backward()
                optimizer1.step()

                optimizer2.zero_grad()
                out2 = model2(batch_x2)
                loss2 = criterion(out2, batch_y)
                loss2.backward()
                optimizer2.step()
     
        else:
            labeled_trainloader1, unlabeled_trainloader1 = update_trainloader(model1, semi_train_loader)
            labeled_trainloader2, unlabeled_trainloader2 = update_trainloader(model2, semi_train_loader)
            Match_train(epoch, model1, optimizer1, labeled_trainloader2, unlabeled_trainloader2, criterion2)
            Match_train(epoch, model2, optimizer2, labeled_trainloader1, unlabeled_trainloader1, criterion2)

        scheduler1.step()
        scheduler2.step()

        test_acc1, test_acc2, avg_acc = evaluate2(test_loader, model1, model2)
        if best_acc < avg_acc:
            best_acc = avg_acc
        logger.info('Noise {} Iter {}: test_acc1={:.4f}, test_acc2={:.4f}, avg_acc={:.4f}'.format(args.noise_type, epoch, test_acc1, test_acc2, avg_acc))

    logger.info('last_acc={:.4f}, best_acc={:.4f}'.format(avg_acc, best_acc))
    return avg_acc, best_acc
    
if __name__ == "__main__":
    tag = f"default"
    results_path = os.path.join('./results/', args.dataset, args.loss, args.noise_type + '_' + args.noise_rate, tag)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    logger = get_logger(results_path + '/result.log')
    logger.info(pprint.pformat(args))
    accs = []
    last_accs = []
    best_accs = []
    for i in range(args.trials):
        last, best = run(args)
        last_accs.append(last)
        best_accs.append(best)
        args.seed += 1
    last_accs = torch.asarray(last_accs)*100
    best_accs = torch.asarray(best_accs)*100

    logger.info(args.dataset+' '+args.loss+' best acc: %.2f±%.2f, last acc: %.2f±%.2f \n' % 
                (best_accs.mean(), best_accs.std(), last_accs.mean(), last_accs.std()))

    

    