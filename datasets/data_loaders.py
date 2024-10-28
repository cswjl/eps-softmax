from torch.utils.data import DataLoader, Dataset
from datasets.data_method1.build_dataset import build_dataset_method1
from datasets.data_method2.build_dataset import build_dataset_method2
from datasets.data_dependent.build_dataset import build_dataset_dependent
from datasets.data_human.bulid_dataset import build_dataset_human
from datasets.transformers import get_transforms
import numpy as np
import torch
import PIL.Image as Image
import os
import torchvision.transforms as transforms

def worker_init_fn(worker_id):
        # Each worker will have its own distinct seed based on the global seed and worker_id
        np.random.seed(123 + worker_id)

def data_loader(args, batch_size):
    train_transform, test_transform = get_transforms(args.dataset)
    if args.noise_type == 'dependent':
        noise_rate = float(args.noise_rate)
        train_dataset, test_dataset = build_dataset_dependent(args.dataset, args.root, args.noise_type, noise_rate, train_transform, test_transform)
    elif args.noise_type == 'asymmetric' or args.noise_type == 'symmetric':
        noise_rate = float(args.noise_rate)
        if args.noise_method == 'method1':
            train_dataset, test_dataset = build_dataset_method1(args.dataset, args.root, args.noise_type, noise_rate, train_transform, test_transform)
        elif args.noise_method == 'method2':
            train_dataset, test_dataset = build_dataset_method2(args.dataset, args.root, args.noise_type, noise_rate, train_transform, test_transform)
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
    
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=16,
                                pin_memory=True,
                                persistent_workers=True,
                                worker_init_fn=worker_init_fn)
        
    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size*2,
                                shuffle=False,
                                num_workers=16,
                                pin_memory=True,
                                persistent_workers=True)
    return train_loader, test_loader

# for mini webvision
class WebVisionDataset:
    def __init__(self, path, file_name='webvision_mini_train', transform=None, target_transform=None):
        self.target_list = []
        self.path = path
        self.load_file(os.path.join(path, file_name))
        self.transform = transform
        self.target_transform = target_transform
        return

    def load_file(self, filename):
        f = open(filename, "r")
        for line in f:
            train_file, label = line.split()
            self.target_list.append((train_file, int(label)))
        f.close()
        return

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, index):
        impath, target = self.target_list[index]
        img = Image.open(os.path.join(self.path, impath)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target

class Clothing1M_Dataset(Dataset):
    def __init__(self, data, labels, root_dir, transform=None, target_transform=None):
        self.train_data = np.array(data)
        self.train_labels = np.array(labels)
        self.root_dir = root_dir
        self.length = len(self.train_labels)

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        self.target_transform = target_transform
        print("NewDataset length:", self.length)

    def __getitem__(self, index):
        img_paths, target = self.train_data[index], self.train_labels[index]

        img_paths = os.path.join(self.root_dir, img_paths)
        img = Image.open(img_paths).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def getData(self):
        return self.train_data, self.train_labels

def webvision_loader(args, batch_size):
    train_transform, test_transform = get_transforms('webvision')
    web_train_dataset = WebVisionDataset(path=args.root,
                                file_name='webvision_mini_train.txt',
                                transform=train_transform)
    web_test_dataset = WebVisionDataset(path=args.root,
                                    file_name='webvision_mini_val.txt',
                                    transform=test_transform)
    
    img_test_set = WebVisionDataset(path='./datasets',
                                file_name='imgnet_val.txt',
                                transform=test_transform)

    web_train_loader = DataLoader(dataset=web_train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True,
                                persistent_workers=True,
                                worker_init_fn=worker_init_fn
                                )
    web_test_loader = DataLoader(dataset=web_test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True,
                                persistent_workers=True,
                            )
    
    img_test_loader = DataLoader(dataset=img_test_set,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True,
                                persistent_workers=True)

    return web_train_loader, web_test_loader, img_test_loader

# for clothing1m
def target_transform(label):
    label = np.array(label, dtype=np.int32)
    target = torch.from_numpy(label).long()
    return target

def clothing1m_loader(args, batch_size):
    train_transform, test_transform = get_transforms('clothing1m')
    kvDic = np.load(args.root + '/Clothing1m-data.npy', allow_pickle=True).item()
    original_train_data = kvDic['train_data']
    original_train_labels = kvDic['train_labels']
    shuffle_index = np.arange(len(original_train_labels), dtype=int)
    np.random.shuffle(shuffle_index)
    original_train_data = original_train_data[shuffle_index]
    original_train_labels = original_train_labels[shuffle_index]
    train_dataset = Clothing1M_Dataset(original_train_data, original_train_labels, args.root, train_transform, target_transform)
    test_data = kvDic['test_data']
    test_labels = kvDic['test_labels']
    test_dataset = Clothing1M_Dataset(test_data, test_labels, args.root, test_transform, target_transform)
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True,
                                persistent_workers=True,
                                worker_init_fn=worker_init_fn
                                )
    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True,
                                persistent_workers=True
                            )
    return train_loader, test_loader