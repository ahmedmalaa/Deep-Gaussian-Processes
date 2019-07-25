

from __future__ import absolute_import, division, print_function

import argparse
import logging
import numpy as np

import pyro
from pyro.contrib.examples.util import get_data_loader

from models.DGP import DeepGP


def test(test_loader, gpmodule):
    correct = 0
    for data, target in test_loader:

        data = data.reshape(-1, 784)
        pred = gpmodule(data)
        # compare prediction and target to count accuaracy
        correct += pred.eq(target).long().cpu().sum().item()

    print("\nTest set: Accuracy: {}/{} ({:.2f}%)\n"
          .format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))



def main(args):

    train_loader = get_data_loader(dataset_name='MNIST',
                                   data_dir='~/.data',
                                   batch_size=1000,
                                   is_training_set=True,
                                   shuffle=True)

    test_loader = get_data_loader(dataset_name='MNIST',
                                  data_dir='~/.data',
                                  batch_size=1000,
                                  is_training_set=False,
                                  shuffle=False)
    
    
    X = train_loader.dataset.data.reshape(-1, 784).float() / 255
    y = train_loader.dataset.targets
    
    deepgp = DeepGP(X, y, num_classes=10)
    
    deepgp.train(args.num_epochs, args.num_iters, args.batch_size, args.learning_rate)
    
    test(test_loader, deepgp)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Deep Gaussian Processes on MNIST")

    parser.add_argument("-n", "--num-epochs", default=5, type=int)
    parser.add_argument("-t", "--num-iters", default=60, type=int)
    parser.add_argument("-b", "--batch-size", default=1000, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)

    args = parser.parse_args()
    
    main(args)


# In[ ]:



