from torchvision import datasets
from PIL import Image
import numpy as np
import os
# Fashion_MEAN = [0.2860]
# Fashion_STD = [0.3530]
# KMNIST_MEAN = [0.1904]
# KMNIST_STD = [0.3475]

def get_sym_T(eta, num_classes):
    '''
    eta: noise rate
    '''
    assert (eta >= 0.) and (eta <= 1.)
    
    diag_mask = np.eye(num_classes)
    rest_mask = 1 - diag_mask
    
    T = diag_mask * (1 - eta) \
        + rest_mask * eta / (num_classes - 1)
    
    return T

def get_asym_T_mnist(eta):
    '''
    eta: noise rate
    '''
    assert (eta >= 0.) and (eta <= 1.)
    
    num_classes = 10
    
    T = np.eye(num_classes)
    # 7 -> 1
    T[7, 7], T[7, 1] = 1. - eta, eta
    # 2 -> 7
    T[2, 2], T[2, 7] = 1. - eta, eta
    # 5 <-> 6
    T[5, 5], T[5, 6] = 1. - eta, eta
    T[6, 6], T[6, 5] = 1. - eta, eta
    # 3 -> 8
    T[3, 3], T[3, 8] = 1. - eta, eta
    
    return T

def get_asym_T_cifar10(eta):
    '''
    eta: noise rate
    '''
    assert (eta >= 0.) and (eta <= 1.)
    
    num_classes = 10
    
    T = np.eye(num_classes)
    # truck -> automobile (9 -> 1)
    T[9, 9], T[9, 1] = 1. - eta, eta
    # bird -> airplane (2 -> 0)
    T[2, 2], T[2, 0] = 1. - eta, eta
    # cat <-> dog (3 <-> 5)
    T[3, 3], T[3, 5] = 1. - eta, eta
    T[5, 5], T[5, 3] = 1. - eta, eta
    # deer -> horse (4 -> 7)
    T[4, 4], T[4, 7] = 1. - eta, eta
    
    return T
    
def get_asym_T_cifar100(eta):
    '''
    eta: noise rate
    '''
    assert (eta >= 0.) and (eta <= 1.)
    
    num_classes = 100
    num_superclasses = 20
    num_subclasses = 5

    T = np.eye(num_classes)

    for i in np.arange(num_superclasses):
        # build T for one superclass
        T_superclass = (1. - eta) * np.eye(num_subclasses)
        for j in np.arange(num_subclasses - 1):
            T_superclass[j, j + 1] = eta
        T_superclass[num_subclasses - 1, 0] = eta
        
        init, end = i * num_subclasses, (i + 1) * num_subclasses
        T[init:end, init:end] = T_superclass

    return T

def create_noisy_labels(labels, trans_matrix):
    '''
    create noisy labels from labels and noisy matrix
    '''
    
    if trans_matrix is None:
        raise ValueError('Noisy matrix is None')
    
    num_trans_matrix = trans_matrix.copy()
    labels = labels.copy()
    
    num_classes = len(trans_matrix)
    class_idx = [np.where(np.array(labels) == i)[0]
                 for i in range(num_classes)]
    num_samples_class = [len(class_idx[idx])
                         for idx in range(num_classes)]
    for real_label in range(num_classes):
        for trans_label in range(num_classes):
            num_trans_matrix[real_label][trans_label] = \
                trans_matrix[real_label][trans_label] * num_samples_class[real_label]
    num_trans_matrix = num_trans_matrix.astype(int)

    for real_label in range(num_classes):
        for trans_label in range(num_classes):

            if real_label == trans_label:
                continue

            num_trans = num_trans_matrix[real_label][trans_label]
            if num_trans == 0:
                continue

            trans_samples_idx = np.random.choice(class_idx[real_label],
                                                 num_trans,
                                                 replace=False)
            class_idx[real_label] = np.setdiff1d(class_idx[real_label],
                                                 trans_samples_idx)
            for idx in trans_samples_idx:
                labels[idx] = trans_label
    
    return labels

class MNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, trans_matrix=None):
        super().__init__(root, train, transform, target_transform, download)
        
        self.trans_matrix = trans_matrix
        if self.trans_matrix is not None:
            self.targets = create_noisy_labels(self.targets.numpy(), trans_matrix)
class KMNIST(datasets.KMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, trans_matrix=None):
        super().__init__(root, train, transform, target_transform, download)
        
        self.trans_matrix = trans_matrix
        if self.trans_matrix is not None:
            self.targets = create_noisy_labels(self.targets.numpy(), trans_matrix)
class FashionMNIST(datasets.FashionMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, trans_matrix=None):
        super().__init__(root, train, transform, target_transform, download)
        
        self.trans_matrix = trans_matrix
        if self.trans_matrix is not None:
            self.targets = create_noisy_labels(self.targets.numpy(), trans_matrix)
class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, trans_matrix=None):
        super().__init__(root, train, transform, target_transform, download)
        
        self.trans_matrix = trans_matrix
        if self.trans_matrix is not None:
            self.targets = create_noisy_labels(self.targets, trans_matrix)

class CIFAR100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, trans_matrix=None):
        super().__init__(root, train, transform, target_transform, download)
        
        self.trans_matrix = trans_matrix
        if self.trans_matrix is not None:
            self.targets = create_noisy_labels(self.targets, trans_matrix)

    
def build_dataset_method2(dataset, root, noise_type, noise_rate, train_transform, test_transform):
    if dataset == 'fashionmnist':
        if noise_type == 'symmetric':
            T = get_sym_T(noise_rate, 10)
        elif noise_type == 'asymmetric':
            T = get_asym_T_mnist(noise_rate)
        else:
            raise ValueError('Wrong noise type! Must be sym or asym')

        train_dataset = FashionMNIST(root=root,
                                train=True,
                                transform=train_transform,
                                trans_matrix=T)
        
        test_dataset = FashionMNIST(root=root,
                                train=False,
                                transform=test_transform)
        
    if dataset == 'kmnist':
        if noise_type == 'symmetric':
            T = get_sym_T(noise_rate, 10)
        elif noise_type == 'asymmetric':
            T = get_asym_T_mnist(noise_rate)
        else:
            raise ValueError('Wrong noise type! Must be sym or asym')

        train_dataset = KMNIST(root=root,
                                train=True,
                                transform=train_transform,
                                trans_matrix=T)
        
        test_dataset = KMNIST(root=root,
                                train=False,
                                transform=test_transform)
        
    if dataset == 'mnist':
        if noise_type == 'symmetric':
            T = get_sym_T(noise_rate, 10)
        elif noise_type == 'asymmetric':
            T = get_asym_T_mnist(noise_rate)
        else:
            raise ValueError('Wrong noise type! Must be sym or asym')
    
        train_dataset = MNIST(root=root,
                                train=True,
                                transform=train_transform,
                                trans_matrix=T)
        
        test_dataset = MNIST(root=root,
                                train=False,
                                transform=test_transform)

    if dataset == 'cifar10':
        if noise_type == 'symmetric':
            T = get_sym_T(noise_rate, 10)
        elif noise_type == 'asymmetric':
            T = get_asym_T_cifar10(noise_rate)
        else:
            raise ValueError('Wrong noise type! Must be sym or asym')
        

        train_dataset = CIFAR10(root=root,
                                    train=True,
                                    transform=train_transform,
                                    trans_matrix=T)

        test_dataset = CIFAR10(root=root,
                                    train=False,
                                    transform=test_transform)
        
        return train_dataset, test_dataset

    if dataset == 'cifar100':
        if noise_type == 'symmetric':
            T = get_sym_T(noise_rate, 100)
        elif noise_type == 'asymmetric':
            T = get_asym_T_cifar100(noise_rate)
        else:
            raise ValueError('Wrong noise type! Must be sym or asym')

        train_dataset = CIFAR100(root=root,
                                    train=True,
                                    transform=train_transform,
                                    trans_matrix=T)
        
        test_dataset = CIFAR100(root=root,
                                    train=False,
                                    transform=test_transform)

    return train_dataset, test_dataset

