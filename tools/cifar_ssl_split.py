import os

import click
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100

import mmcv
from mmcv.utils import mkdir_or_exist


@click.command()
@click.option('--data-path', type=str, required=True,
              help='CIFAR10/CIFAR100 dataset path')
@click.option('--dataset-type', type=click.Choice(['CIFAR10', 'CIFAR100']),
              required=True, help='CIFAR10/CIFAR100 dataset type')
@click.option('--num-labeled', type=int, required=True,
              help='num of labeled data num (the data would be balanced across labeld)')
@click.option('--output-path', type=str, required=True,
              help='output path to save labeled and unlabeld idxs')
def main(data_path, dataset_type, num_labeled, output_path):
    params = dict(
        root=data_path,
        train=True,
        download=False
    )
    if dataset_type == 'CIFAR10':
        dataset = CIFAR10(**params)
    elif dataset_type == 'CIFAR100':
        dataset = CIFAR100(**params)
    
    num_classes = len(dataset.classes)
    data, targets = dataset.data, dataset.targets

    label_per_class = num_labeled // num_classes
    labels = np.array(targets)
    labeled_idx = []
    # labeled data included in unlabeled data
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == num_labeled
    
    mkdir_or_exist(output_path)
    mmcv.dump(labeled_idx.tolist(), os.path.join(output_path, 'labeled_idx.pkl'))
    mmcv.dump(unlabeled_idx.tolist(), os.path.join(output_path, 'unlabeled_idx.pkl'))


if __name__ == '__main__':
    main()