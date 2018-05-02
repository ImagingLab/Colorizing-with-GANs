from __future__ import print_function
import os
import argparse


class ModelOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Colorization with GANs')
        parser.add_argument('--dataset', type=str, default='places365',  help='The name of dataset [places365, cifar10] (default: places365)')
        parser.add_argument('--dataset-path', type=str, default='./dataset',  help='dataset path (default: ./dataset)')
        parser.add_argument('--checkpoints-path', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
        parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 30)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
        parser.add_argument('--log-interval', type=int, default=50, metavar='N', help='how many batches to wait before logging training status')
        parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--debug', type=int, default=0, help='debugging (default: 0)')

        self._parser = parser

    def parse(self):
        opt = self._parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
        opt.debug = opt.debug != 0

        args = vars(opt)
        print('\n------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------\n')

        return opt
