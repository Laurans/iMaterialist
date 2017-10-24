# coding: utf8
import argparse
import torch.optim as optim
from classifier_workflow import Experiment
import distutils.util
from prediction_workflow import prediction
import torch

hyperparameters = {
    'batch_size': 16,
    'num_workers': 8,
    'shuffle': True,
    'lr_decay_epoch': 2,
    'num_epochs': 30,
    'optimizer': optim.Adam,
    'optim_params': {
        'lr': 1e-4,
        #'momentum': 0.9
        #'weight_decay':1e-9
    },

}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose model submissions')
    parser.add_argument('--path', type=str, default='', help='Path with model source code')
    parser.add_argument('-ds', '--data_serialized', type=distutils.util.strtobool, default='false')
    parser.add_argument('-pm', '--preprocess_models', type=distutils.util.strtobool, default='true')
    opt = parser.parse_args()
    trained_model = Experiment(opt.path, opt.data_serialized, opt.preprocess_models, hyperparameters).train()
    #trained_model = torch.load('../models/best_model')
    #prediction(trained_model)
