import torch.nn as nn
from losses import *

MNIST_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(num_classes=10),
    "MSE": MSELoss(num_classes=10),
    "GCE": GCELoss(num_classes=10),
    "SCE": SCELoss(num_classes=10, a=0.01, b=1),
    "RCE": RCELoss(num_classes=10),
    "NFL": NormalizedFocalLoss(gamma=0.5, num_classes=10),
    "NCE": NCELoss(num_classes=10),
    "AEL": AExpLoss(num_classes=10, a=3.5),
    "AUL": AUELoss(num_classes=10, a=3, q=0.1),
    "AGCE": AGCELoss(num_classes=10, a=4, q=0.2),
    "NFLandRCE": NFLandRCE(alpha=1, beta=10, num_classes=10, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=1, beta=10, num_classes=10),
    "NCEandRCE": NCEandRCE(alpha=1, beta=10, num_classes=10),
    "NCEandAGCE": NCEandAGCE(alpha=0, beta=1, num_classes=10, a=4, q=0.2),
    "NCEandAUL": NCEandAUE(alpha=0, beta=1, num_classes=10, a=3, q=0.1),
    "NCEandAEL": NCEandAEL(alpha=0, beta=1, num_classes=10, a=3.5),
    "LDR": LDRLoss_V1(threshold=0.1, Lambda=10),
}

CIFAR10_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(num_classes=10),
    "MSE": MSELoss(num_classes=10),
    "GCE": GCELoss(num_classes=10),
    "SCE": SCELoss(num_classes=10, a=0.1, b=1),
    "RCE": RCELoss(num_classes=10),
    "NFL": NormalizedFocalLoss(gamma=0.5, num_classes=10),
    "NCE": NCELoss(num_classes=10),
    "AEL": AExpLoss(num_classes=10, a=2.5),
    "AUL": AUELoss(num_classes=10, a=5.5, q=3),
    "AGCE": AGCELoss(num_classes=10, a=0.6, q=0.6),
    "NFLandRCE": NFLandRCE(alpha=1, beta=1, num_classes=10, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=1, beta=1, num_classes=10),
    "NCEandRCE": NCEandRCE(alpha=1, beta=1, num_classes=10),
    "NCEandAGCE": NCEandAGCE(alpha=1, beta=4, num_classes=10, a=6, q=1.5),
    "NCEandAUL": NCEandAUE(alpha=1, beta=4, num_classes=10, a=6.3, q=1.5),
    "NCEandAEL": NCEandAEL(alpha=1, beta=4, num_classes=10, a=5),
    "LDR": LDRLoss_V1(threshold=0.1, Lambda=10),
}

CIFAR100_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(num_classes=100),
    "MSE": MSELoss(num_classes=100),
    "GCE": GCELoss(num_classes=100, q=0.7),
    "SCE": SCELoss(num_classes=100, a=6, b=0.1),
    "RCE": RCELoss(num_classes=100),
    "NFL": NormalizedFocalLoss(gamma=0.5, num_classes=100),
    "NCE": NCELoss(num_classes=100),
    "AEL": AExpLoss(num_classes=100, a=2.5),
    "AUL": AUELoss(num_classes=100, a=5.5, q=3),
    "AGCE": AGCELoss(num_classes=100, a=0.6, q=0.6),
    "NFLandRCE": NFLandRCE(alpha=10, beta=0.1, num_classes=100, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=10, beta=0.1, num_classes=100),
    "NCEandRCE": NCEandRCE(alpha=10, beta=0.1, num_classes=100),
    "NCEandAGCE": NCEandAGCE(alpha=10, beta=0.1, num_classes=100, a=1.8, q=3),
    "NCEandAUL": NCEandAUE(alpha=10, beta=0.015, num_classes=100, a=6, q=3),
    "NCEandAEL": NCEandAEL(alpha=10, beta=0.1, num_classes=100, a=1.5),
    "LDR": LDRLoss_V1(threshold=0.1, Lambda=1),
}
WEBVISION_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "GCE": GCELoss(num_classes=50, q=0.7),
    "SCE": SCELoss(num_classes=50, a=10, b=1),
    "AGCE": AGCELoss(num_classes=50, a=1e-5, q=0.5),
    "NCEandRCE": NCEandRCE(alpha=50, beta=0.1, num_classes=50),
    "NCEandAGCE": NCEandAGCE(alpha=50, beta=0.1, num_classes=50, a=2.5, q=3),
    "LDR": LDRLoss_V1(threshold=0.1, Lambda=1),
}
CLOTHING_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "GCE": GCELoss(num_classes=14, q=0.6),
    "SCE": SCELoss(num_classes=14, a=10, b=1),
    "AGCE": AGCELoss(num_classes=14, a=1e-5, q=0.5),
    "NCEandRCE": NCEandRCE(alpha=10, beta=1, num_classes=14),
    "NCEandAGCE": NCEandAGCE(alpha=50, beta=0.1, num_classes=14, a=2.5, q=3),
    "LDR": LDRLoss_V1(threshold=0.1, Lambda=1),
}

def get_loss_config(args):
    if args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'webvision':
        num_classes = 50
    elif args.dataset == 'clothing1m':
        num_classes = 14
    elif args.dataset == 'cifar10':
        num_classes = 10

    if args.loss == 'ECEandMAE':
        if args.dataset == 'cifar10':
            if args.noise_type == 'symmetric':  return ECEandMAE(alpha=0.01, beta=5, m=1e5, num_classes=num_classes)
            if args.noise_type == 'asymmetric': return ECEandMAE(alpha=0.02, beta=5, m=1e3, num_classes=num_classes)
            if args.noise_type == 'dependent':  return ECEandMAE(alpha=0.045, beta=10, m=1e5, num_classes=num_classes)
        if args.dataset == 'cifar100':
            if args.noise_type == 'symmetric' or args.noise_type == 'dependent':
                if args.noise_rate == '0':  return ECEandMAE(alpha=0.1, beta=1, m=1e4, num_classes=num_classes)
                if args.noise_rate == '0.2':  return ECEandMAE(alpha=0.05, beta=1, m=1e4, num_classes=num_classes)
                if args.noise_rate == '0.4':  return ECEandMAE(alpha=0.03, beta=1, m=1e4, num_classes=num_classes)
                if args.noise_rate == '0.6':  return ECEandMAE(alpha=0.00125, beta=1, m=1e4, num_classes=num_classes)
                if args.noise_rate == '0.8':  return ECEandMAE(alpha=0.0075, beta=1, m=1e4, num_classes=num_classes)
            if args.noise_type == 'asymmetric':
                if args.noise_rate == '0.1':  return ECEandMAE(alpha=0.015, beta=1, m=1e2, num_classes=num_classes)
                if args.noise_rate == '0.2':  return ECEandMAE(alpha=0.007, beta=1, m=1e2, num_classes=num_classes)
                if args.noise_rate == '0.3':  return ECEandMAE(alpha=0.005, beta=1, m=1e2, num_classes=num_classes)
                if args.noise_rate == '0.4':  return ECEandMAE(alpha=0.004, beta=1, m=1e2, num_classes=num_classes)
           
    if args.loss == 'EFLandMAE':
        if args.dataset == 'cifar10':
            if args.noise_type == 'symmetric':  return EFLandMAE(alpha=0.01, beta=5, m=1e5, num_classes=num_classes)
            if args.noise_type == 'asymmetric': return EFLandMAE(alpha=0.02, beta=5, m=1e3, num_classes=num_classes)
            if args.noise_type == 'dependent':  return EFLandMAE(alpha=0.045, beta=10, m=1e5, num_classes=num_classes)
        if args.dataset == 'cifar100':
            if args.noise_type == 'symmetric' or args.noise_type == 'dependent':
                if args.noise_rate == '0':  return EFLandMAE(alpha=0.1, beta=1, m=1e4, num_classes=num_classes)
                if args.noise_rate == '0.2':  return EFLandMAE(alpha=0.05, beta=1, m=1e4, num_classes=num_classes)
                if args.noise_rate == '0.4':  return EFLandMAE(alpha=0.03, beta=1, m=1e4, num_classes=num_classes)
                if args.noise_rate == '0.6':  return EFLandMAE(alpha=0.00125, beta=1, m=1e4, num_classes=num_classes)
                if args.noise_rate == '0.8':  return EFLandMAE(alpha=0.0075, beta=1, m=1e4, num_classes=num_classes)
            if args.noise_type == 'asymmetric':
                if args.noise_rate == '0.1':  return EFLandMAE(alpha=0.015, beta=1, m=1e2, num_classes=num_classes)
                if args.noise_rate == '0.2':  return EFLandMAE(alpha=0.007, beta=1, m=1e2, num_classes=num_classes)
                if args.noise_rate == '0.3':  return EFLandMAE(alpha=0.005, beta=1, m=1e2, num_classes=num_classes)
                if args.noise_rate == '0.4':  return EFLandMAE(alpha=0.004, beta=1, m=1e2, num_classes=num_classes)


    if args.loss == 'CEandLC':
        if args.dataset == 'cifar10':
            if args.noise_type == 'symmetric':
                if args.noise_rate < 0.5:
                    return CEandLC(delta=1)
                else:
                    return CEandLC(delta=1.5)
            else:
                return CEandLC(delta=2.5)
        elif args.dataset == 'cifar100':
            if args.noise_type == 'symmetric':
                return CEandLC(delta=0.5)
            else:
                return CEandLC(delta=2.5)

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
 
