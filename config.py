import torch.nn as nn
from losses import *

MNIST_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(),
    "MSE": MSELoss(),
    "GCE": GCELoss(),
    "SCE": SCELoss(a=0.01, b=1),
    "RCE": RCELoss(),
    "NFL": NormalizedFocalLoss(),
    "NCE": NCELoss(),
    "AEL": AExpLoss(a=3.5),
    "AUL": AUELoss(a=3, q=0.1),
    "AGCE": AGCELoss(a=4, q=0.2),
    "NFLandRCE": NFLandRCE(alpha=1, beta=10),
    "NCEandMAE": NCEandMAE(alpha=1, beta=10),
    "NCEandRCE": NCEandRCE(alpha=1, beta=10),
    "NCEandAGCE": NCEandAGCE(alpha=0, beta=1, a=4, q=0.2),
    "NCEandAUL": NCEandAUE(alpha=0, beta=1, a=3, q=0.1),
    "NCEandAEL": NCEandAEL(alpha=0, beta=1, a=3.5),
    "LDR": LDRLoss_V1(threshold=0.1, Lambda=10),
    "NCEandNNCE": NCEandNNCE(alpha=1, beta=1),
    "NFLandNNFL": NFLandNNFL(alpha=1, beta=1),
    "NNCE": NormalizedNegativeCrossEntropy() 
}

CIFAR10_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(),
    "MSE": MSELoss(),
    "GCE": GCELoss(),
    "SCE": SCELoss(a=0.1, b=1),
    "RCE": RCELoss(),
    "NFL": NormalizedFocalLoss(),
    "NCE": NCELoss(),
    "AEL": AExpLoss(a=2.5),
    "AUL": AUELoss(a=5.5, q=3),
    "AGCE": AGCELoss(a=0.6, q=0.6),
    "NFLandRCE": NFLandRCE(alpha=1, beta=1, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=1, beta=1),
    "NCEandRCE": NCEandRCE(alpha=1, beta=1),
    "NCEandAGCE": NCEandAGCE(alpha=1, beta=4, a=6, q=1.5),
    "NCEandAUL": NCEandAUE(alpha=1, beta=4, a=6.3, q=1.5),
    "NCEandAEL": NCEandAEL(alpha=1, beta=4, a=5),
    "LDR": LDRLoss_V1(threshold=0.1, Lambda=10),
    "NCEandNNCE": NCEandNNCE(alpha=5, beta=5),
    "NFLandNNFL": NFLandNNFL(alpha=5, beta=5),
    "NNCE": NormalizedNegativeCrossEntropy() 
}

CIFAR100_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(),
    "MSE": MSELoss(),
    "GCE": GCELoss(q=0.7),
    "SCE": SCELoss(a=6, b=0.1),
    "RCE": RCELoss(),
    "NFL": NormalizedFocalLoss(),
    "NCE": NCELoss(),
    "AEL": AExpLoss(a=2.5),
    "AUL": AUELoss(a=5.5, q=3),
    "AGCE": AGCELoss(a=0.6, q=0.6),
    "NFLandRCE": NFLandRCE(alpha=10, beta=0.1, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=10, beta=0.1),
    "NCEandRCE": NCEandRCE(alpha=10, beta=0.1),
    "NCEandAGCE": NCEandAGCE(alpha=10, beta=0.1, a=1.8, q=3),
    "NCEandAUL": NCEandAUE(alpha=10, beta=0.015, a=6, q=3),
    "NCEandAEL": NCEandAEL(alpha=10, beta=0.1, a=1.5),
    "LDR": LDRLoss_V1(threshold=0.1, Lambda=1),
    "NCEandNNCE": NCEandNNCE(alpha=10, beta=1),
    "NNCE": NormalizedNegativeCrossEntropy(), 
}
WEBVISION_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "GCE": GCELoss(q=0.7),
    "SCE": SCELoss(a=10, b=1),
    "AGCE": AGCELoss(a=1e-5, q=0.5),
    "NCEandRCE": NCEandRCE(alpha=50, beta=0.1),
    "NCEandAGCE": NCEandAGCE(alpha=50, beta=0.1, a=2.5, q=3),
    "NCEandNNCE": NCEandNNCE(alpha=20, beta=1),
    "NFLandNNFL": NFLandNNFL(alpha=20, beta=1),
}
CLOTHING_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "GCE": GCELoss(q=0.6),
    "SCE": SCELoss(a=10, b=1),
    "AGCE": AGCELoss(a=1e-5, q=0.5),
    "LDR": LDRLoss_V1(threshold=0.1, Lambda=1),
    "NCEandRCE": NCEandRCE(alpha=10, beta=1),
    "NCEandAGCE": NCEandAGCE(alpha=50, beta=0.1, a=2.5, q=3),
    "NCEandNNCE": NCEandNNCE(alpha=5, beta=0.1),
    "NFLandNNFL": NFLandNNFL(alpha=5, beta=0.1),
}

def get_eps_softmax(args):
    if args.loss in ['ECEandMAE', 'EFLandMAE']:
        if args.dataset == 'cifar10':
            if args.noise_type == 'symmetric':  alpha, beta, m = 0.01, 5, 1e5
            elif args.noise_type == 'asymmetric': alpha, beta, m = 0.02, 5, 1e3
            elif args.noise_type == 'dependent':  alpha, beta, m = 0.045, 10, 1e5
            else: raise ValueError('No default parameter for this case, set yourself!')
        elif args.dataset == 'cifar100':
            if args.noise_type == 'symmetric' or args.noise_type == 'dependent':
                if args.noise_rate in ['0', '0.0']:  alpha, beta, m = 0.1, 1, 1e4
                elif args.noise_rate == '0.2':  alpha, beta, m = 0.05, 1, 1e4
                elif args.noise_rate == '0.4':  alpha, beta, m = 0.03, 1, 1e4
                elif args.noise_rate == '0.6':  alpha, beta, m = 0.00125, 1, 1e4
                elif args.noise_rate == '0.8':   alpha, beta, m = 0.0075, 1, 1e4
                else: raise ValueError('No default parameter for this case, set yourself!')
            elif args.noise_type == 'asymmetric':
                if args.noise_rate == '0.1':  alpha, beta, m = 0.015, 1, 1e2
                elif args.noise_rate == '0.2':  alpha, beta, m = 0.007, 1, 1e2
                elif args.noise_rate == '0.3':  alpha, beta, m = 0.005, 1, 1e2
                elif args.noise_rate == '0.4':  alpha, beta, m = 0.004, 1, 1e2
                else: raise ValueError('No default parameter for this case, set yourself!')
        elif args.dataset == 'webvision': alpha, beta, m = 0.015, 0.3, 1e3
        elif args.dataset == 'clothing1m': alpha, beta, m = 0.012, 0.1, 1e3
        else: raise ValueError('No default parameter for this case, set yourself!')

        if args.loss == 'ECEandMAE':
            return ECEandMAE(alpha=alpha, beta=beta, m=m)
        else:
            return EFLandMAE(alpha=alpha, beta=beta, m=m)

def get_ce_and_lc(args):
    if args.dataset == 'cifar10':
        if args.noise_type == 'symmetric':
            if float(args.noise_rate) < 0.5:
                return CEandLC(delta=1)
            else:
                return CEandLC(delta=1.5)
        elif args.noise_type in ['asymmetric', 'human']:
            return CEandLC(delta=2.5)
        elif args.noise_type == 'dependent':
            return CEandLC(delta=2)
    elif args.dataset == 'cifar100':
        if args.noise_type == 'asymmetric':
            return CEandLC(delta=2.5)
        else:
            return CEandLC(delta=0.5)
    elif args.dataset == 'webvision':
        return CEandLC(delta=1.2)
    else: raise ValueError('No default parameter for this case, set yourself!')



def get_loss_config(args):
    if args.loss in ['ECEandMAE', 'EFLandMAE']:
        return get_eps_softmax(args)
    if args.loss == 'CEandLC':
        return get_ce_and_lc(args)
    if args.dataset == 'mnist':
        return MNIST_CONFIG[args.loss]
    if args.dataset == 'cifar10':
        return CIFAR10_CONFIG[args.loss]
    if args.dataset == 'cifar100':
        return CIFAR100_CONFIG[args.loss]
    if args.dataset == 'webvision':
        return WEBVISION_CONFIG[args.loss]
    if args.dataset == 'clothing1m':
        return CLOTHING_CONFIG[args.loss]

def get_weight_decay_config(args):
    if args.dataset == 'mnist':
        l1_weight_decay = 0
        l2_weight_decay = 1e-3
    elif args.dataset == 'cifar10':
        l1_weight_decay = 0
        l2_weight_decay = 1e-4
    elif args.dataset == 'cifar100':
        l1_weight_decay = 0
        l2_weight_decay = 1e-5
    elif args.dataset == 'webvision':
        l1_weight_decay = 0
        l2_weight_decay = 3e-5
    elif args.dataset == 'clothing1m':
        l1_weight_decay = 0
        l2_weight_decay = 1e-3
    else:
        raise NotImplementedError
    return l1_weight_decay, l2_weight_decay
       