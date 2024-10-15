from .cifar import CIFAR10, CIFAR100

def build_dataset_human(dataset, root, noise_type, noise_path, train_transform, test_transform):
    if dataset == 'cifar10':
        train_dataset = CIFAR10(root=root,
                                download=True,  
                                train=True, 
                                transform=train_transform,
                                noise_type=noise_type,
                                noise_path=noise_path, is_human=True
                           )
        test_dataset = CIFAR10(root=root,
                                download=False,  
                                train=False, 
                                transform=test_transform,
                                noise_type=noise_type
                          )
    elif dataset == 'cifar100':
        train_dataset = CIFAR100(root=root,
                                download=True,  
                                train=True, 
                                transform=train_transform,
                                noise_type=noise_type,
                                noise_path = noise_path, is_human=True
                            )
        test_dataset = CIFAR100(root=root,
                                download=False,  
                                train=False, 
                                transform=test_transform,
                                noise_type=noise_type
                            )
    return train_dataset, test_dataset








