import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_main_semi(dataset, noise_type, noise_rate):
    spec = importlib.util.spec_from_file_location('main_semi_test', REPO_ROOT / 'main_semi.py')
    module = importlib.util.module_from_spec(spec)
    argv = ['main_semi.py', '--dataset', dataset, '--noise_type', noise_type,
            '--noise_rate', noise_rate]
    with patch.object(sys, 'argv', argv), patch.dict('os.environ'):
        spec.loader.exec_module(module)
    return module


class SemiNoiseTests(unittest.TestCase):
    def exercise_warmup(self, module, builder_name, labels, clean=False):
        """Run a real CPU warm-up epoch with tiny images and lightweight models."""
        labels = np.asarray(labels)
        images = np.zeros((len(labels), 32, 32, 3), dtype=np.uint8)
        other_labels = (labels + 1) % module.num_classes
        train_dataset = SimpleNamespace(
            train_data=images,
            train_labels=labels if clean else other_labels,
            train_noisy_labels=other_labels if clean else labels,
        )
        test_dataset = TensorDataset(torch.zeros(len(labels), 3, 32, 32),
                                     torch.as_tensor(labels))
        models = []
        loaders = []

        def build_model(num_classes):
            self.assertEqual(num_classes, module.num_classes)
            model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, num_classes))
            models.append(model)
            return model

        def build_loader(**kwargs):
            kwargs.update(num_workers=0, persistent_workers=False, pin_memory=False)
            loader = DataLoader(**kwargs)
            loaders.append(loader)
            return loader

        with patch.object(module, builder_name, return_value=(train_dataset, test_dataset)) as builder, \
                patch.object(module, 'ResNet34', side_effect=build_model), \
                patch.object(module, 'DataLoader', side_effect=build_loader), \
                patch.object(module, 'tqdm', side_effect=lambda iterable, **kwargs: iterable), \
                patch.object(module.torch.cuda, 'is_available', return_value=False), \
                patch.multiple(module, device='cpu', epochs=1, batch_size=len(labels)), \
                patch.object(module, 'logger', Mock(), create=True):
            last_acc, best_acc = module.run(module.args)

        self.assertEqual(len(models), 2)
        self.assertEqual(len(loaders), 3)
        for loader in loaders[:2]:
            np.testing.assert_array_equal(loader.dataset.data, images)
            np.testing.assert_array_equal(loader.dataset.targets, labels)
        self.assertIs(loaders[2].dataset, test_dataset)
        for model in models:
            self.assertTrue(all(p.grad is not None for p in model.parameters()))
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        self.assertTrue(0 <= last_acc <= best_acc <= 1)
        return builder

    def test_dependent_noise_for_both_datasets_and_all_bundled_rates(self):
        for dataset in ('cifar10', 'cifar100'):
            for rate in ('0.2', '0.4', '0.6'):
                with self.subTest(dataset=dataset, rate=rate):
                    module = load_main_semi(dataset, 'dependent', rate)
                    label_file = REPO_ROOT / 'datasets/data_dependent/config' / f'{dataset}_dependent_{rate}.json'
                    labels = np.asarray(json.loads(label_file.read_text()))
                    builder = self.exercise_warmup(module, 'build_dataset_dependent', labels[:8])
                    builder.assert_called_once_with(
                        dataset, module.args.root, 'dependent', float(rate),
                        module.train_transform, module.test_transform,
                    )
                    self.assertTrue(module.args.root.endswith('/' + dataset))
                    # Ensure the default selection count works with actual noisy class sizes.
                    losses = torch.linspace(0, 1, len(labels))
                    good, bad = module.select_samples(losses, labels, module.k)
                    self.assertTrue(good)
                    self.assertTrue(bad)
                    self.assertFalse(set(good) & set(bad))
                    self.assertEqual(set(good) | set(bad), set(range(len(labels))))
                    self.assertLessEqual(np.bincount(labels[good]).max(), module.k)

    def test_human_noise_keeps_clean_and_noisy_label_selection(self):
        cases = (('cifar10', 'worst'), ('cifar10', 'clean'),
                 ('cifar100', 'noisy100'), ('cifar100', 'clean100'))
        for dataset, variant in cases:
            with self.subTest(dataset=dataset, variant=variant):
                module = load_main_semi(dataset, 'human', variant)
                clean = variant in ('clean', 'clean100')
                self.exercise_warmup(module, 'build_dataset_human', np.arange(8), clean=clean)


if __name__ == '__main__':
    unittest.main()
