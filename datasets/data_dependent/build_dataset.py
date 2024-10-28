from datasets.data_dependent.cifar import CIFAR10, CIFAR100

def build_dataset_dependent(dataset, root, noise_type, noise_rate, train_transform, test_transform):
    if dataset == 'cifar10':
        train_dataset = CIFAR10(root=root,
                        download=True,
                        dataset_type="train",
                        transform=train_transform,
                        noise_type=noise_type,
                        noise_rate=noise_rate
                        )
        
        test_dataset = CIFAR10(root=root,
                        download=True,
                        dataset_type="test",
                        transform=test_transform,
                        noise_type=noise_type,
                        noise_rate=noise_rate
                        )
    elif dataset == 'cifar100':
        train_dataset = CIFAR100(root=root,
                        download=True,
                        dataset_type="train",
                        transform=train_transform,
                        noise_type=noise_type,
                        noise_rate=noise_rate
                        )
        
        test_dataset = CIFAR100(root=root,
                        download=True,
                        dataset_type="test",
                        transform=test_transform,
                        noise_type=noise_type,
                        noise_rate=noise_rate
                        )

    return train_dataset, test_dataset





