from torchvision import datasets
from numpy.testing import assert_array_almost_equal
import numpy as np

import logging

logger = logging.getLogger(__name__)

def build_for_cifar100(size, noise):
    """ random flip between two random classes.
    """
    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise

    # adjust last row
    P[size-1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class



class NoisyMNIST(datasets.MNIST):
    def __init__(self, root, transform=None, train=True, target_transform=None, download=True, noise_rate=0.0, is_asym=False):
        super(NoisyMNIST, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.targets = self.targets.numpy()
        if is_asym:
            P = np.eye(10)
            n = noise_rate

            # 7 -> 1
            P[7, 7], P[7, 1] = 1. - n, n
            # 2 -> 7
            P[2, 2], P[2, 7] = 1. - n, n
            # 5 <-> 6
            P[5, 5], P[5, 6] = 1. - n, n
            P[6, 6], P[6, 5] = 1. - n, n
            # 3 -> 8
            P[3, 3], P[3, 8] = 1. - n, n

            y_train_noisy = multiclass_noisify(self.targets, P=P)
            actual_noise = (y_train_noisy != self.targets).mean()
            assert actual_noise > 0.0
            logger.info('Actual noise %.2f' % actual_noise)
            self.targets = y_train_noisy
        else:
            P = np.ones((10, 10))
            n = noise_rate
            P = (n / (10 - 1)) * P
            if n > 0.0:
                n_samples = len(self.targets)
                P[0, 0] = 1. - n
                for i in range(1, 10 - 1):
                    P[i, i] = 1. - n
                P[10 - 1, 10 - 1] = 1. - n

                y_train_noisy = multiclass_noisify(self.targets, P=P)
                actual_noise = (y_train_noisy != self.targets).mean()
                assert actual_noise > 0.0
                logger.info('Actual noise %.2f' % actual_noise)
                self.targets = y_train_noisy
        logger.info('Print noisy label generation statistics:')
        for i in range(10):
            n_noisy = np.sum(np.array(self.targets) == i)
            logger.info("Noisy class %s, has %s samples." % (i, n_noisy))
        return

class NoisyCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, noise_rate=0.0, is_asym=False):
        super(NoisyCIFAR10, self).__init__(root, download=download, transform=transform, target_transform=target_transform)
        self.download = download
        if is_asym:
            # automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
            source_class = [9, 2, 3, 5, 4]
            target_class = [1, 0, 5, 3, 7]
            for s, t in zip(source_class, target_class):
                cls_idx = np.where(np.array(self.targets) == s)[0]
                n_noisy = int(noise_rate * cls_idx.shape[0])
                noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
                for idx in noisy_sample_index:
                    self.targets[idx] = t
            return
        elif noise_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(noise_rate * n_samples)
            logger.info("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
            class_noisy = int(n_noisy / 10)
            noisy_idx = []
            for d in range(10):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                logger.info("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=10, current_class=self.targets[i])
            logger.info(str(len(noisy_idx)))
            logger.info("Print noisy label generation statistics:")
            for i in range(10):
                n_noisy = np.sum(np.array(self.targets) == i)
                logger.info("Noisy class %s, has %s samples." % (i, n_noisy))
            return

class NoisyCIFAR100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, noise_rate=0.0, is_asym=False):
        super(NoisyCIFAR100, self).__init__(root, download=download, train=train, transform=transform, target_transform=target_transform)
        self.download = download
        if is_asym:
            nb_classes = 100
            P = np.eye(nb_classes)
            n = noise_rate
            nb_superclasses = 20
            nb_subclasses = 5
            if n > 0.0:
                for i in range(nb_superclasses):
                    init, end = i * nb_subclasses, (i + 1) * nb_subclasses
                    P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

                    y_train_noisy = multiclass_noisify(np.array(self.targets), P=P)
                    actual_noise = (y_train_noisy != np.array(self.targets)).mean()
                assert actual_noise > 0.0
                logger.info('Actual noise: %.2f' % actual_noise)
                self.targets = y_train_noisy.tolist()
            return
        elif noise_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(noise_rate * n_samples)
            logger.info("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(100)]
            class_noisy = int(n_noisy / 100)
            noisy_idx = []
            for d in range(100):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                logger.info("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=100, current_class=self.targets[i])
            logger.info(str(len(noisy_idx)))
            logger.info("Print noisy label generation statistics:")
            for i in range(100):
                n_noisy = np.sum(np.array(self.targets) == i)
                logger.info("Noisy class %s, has %s samples." % (i, n_noisy))
            return


def build_dataset_method1(dataset, root, noise_type, noise_rate, train_transform, test_transform):
    if noise_type == 'asymmetric':
        is_asym = True
    else:
        is_asym = False
    if dataset == 'mnist': 
        train_dataset = NoisyMNIST(root=root,
                                       train=True,
                                       transform=train_transform,
                                       download=True,
                                       noise_rate=noise_rate,
                                       is_asym=is_asym,
                                       )
        test_dataset = datasets.MNIST(root=root,
                                        train=False,
                                        transform=test_transform,
                                        download=True
                                        )
    if dataset == 'cifar10':
        train_dataset = NoisyCIFAR10(root=root,
                                        train=True,
                                        transform=train_transform,
                                        download=True,
                                        is_asym=is_asym,
                                        noise_rate=noise_rate)
        
        test_dataset = datasets.CIFAR10(root=root,
                                        train=False,
                                        transform=test_transform,
                                        download=True)
    if dataset == 'cifar100':
        train_dataset = NoisyCIFAR100(root=root,
                                        train=True,
                                        transform=train_transform,
                                        download=True,
                                        is_asym=is_asym,
                                        noise_rate=noise_rate)
        
        test_dataset = datasets.CIFAR100(root=root,
                                        train=False,
                                        transform=test_transform,
                                        download=True)
    return train_dataset, test_dataset