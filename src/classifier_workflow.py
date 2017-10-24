# coding: utf8
import copy
import pickle
from datetime import datetime
from importlib import import_module
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.autograd import Variable
from logger import Logger, EarlyStopping
import numpy as np
import ipdb

use_gpu = torch.cuda.is_available()
visdom = True

class Experiment():
    def __init__(self, module_path, pre_serialized, preprocess_model, hyperparams):
        if module_path:
            module_path += '.'

        self.hyperparams = hyperparams

        self.model_creator = import_module(module_path + 'model_creator')
        data_preprocessor = import_module(module_path + 'data_preprocessor')
        if visdom:
            self.logger = Logger(module_path)


        print('---> Loading data <---')
        if pre_serialized:
            self.dset_loaders, self.dset_sizes, self.infos = data_preprocessor.load_serialized_data()
        else:
            self.dset_loaders, self.dset_sizes, self.infos = data_preprocessor.load_data_from_scratch(hyperparams,
                                                                                                      preprocess_model)

        print('Dataset size:', self.dset_sizes)

    def train(self):
        print('---> Building graph model <---')
        model, code_src = self.model_creator.build_model(self.infos, use_gpu)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.hyperparams["optimizer"](model.get_parameters(), **self.hyperparams['optim_params'])

        print('---> Training and evaluating <---')
        trained_model, results = self.train_model(model)

        print('---> Save model, hyperparameters and results <---')
        filename = '../models/{}_{:%m%d-%H%M}'.format(self.model_creator.get_model_name(), datetime.now())
        torch.save(trained_model.state_dict(), filename)

        results['code_src'] = code_src
        results.update(self.hyperparams)
        results['filename'] = filename
        pickle.dump(results, open(filename + '_results.pkl', 'wb'))
        return trained_model

    def train_model(self, model):
        num_epochs = self.hyperparams['num_epochs']
        since = datetime.now()

        best_model = model
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    optimizer = lr_scheduler(self.optimizer, epoch, init_lr=self.hyperparams['optim_params']['lr'],
                                            lr_decay_epoch=self.hyperparams['lr_decay_epoch'])
                    model.train(True)
                else:
                    model.train(False)

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                i = 0
                for data in tqdm(self.dset_loaders[phase], ncols=50, ascii=True):
                    # get the inputs
                    inputs_tensors, labels_tensors = data
                    # wrap them in Variable
                    inputs = list(map(lambda x: Variable(x.cuda()),inputs_tensors))
                    labels = list(map(lambda x: Variable(x.cuda()), labels_tensors))

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    # import ipdb; ipdb.set_trace()

                    _, preds = torch.max(outputs[0].data, 1)
                    loss = multicriterion(outputs, labels)
                    # loss = self.criterion(outputs, labels)

                    # import ipdb; ipdb.set_trace()

                    #  backward + optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # print statistics
                    running_loss += loss.data[0]
                    corrects = torch.sum(preds == labels[0].data)
                    if visdom:
                        self.logger.log_metrics('Acc_{}_{}'.format(phase,epoch), phase, np.array([i]), np.array([corrects/preds.size(0)]))
                    print(" Epoch {} | Acc iteration: {:.4f}".format(epoch,corrects/preds.size(0)))
                    running_corrects += corrects
                    i += 1

                epoch_loss = running_loss / self.dset_sizes[phase]
                epoch_acc = running_corrects / self.dset_sizes[phase]
                
                if visdom:
                    self.logger.log_metrics('Loss', phase, np.array([epoch]), np.array([epoch_loss]))
                    self.logger.log_metrics('Acc', phase, np.array([epoch]), np.array([epoch_acc]))

                print('{} | Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model)
                    torch.save(best_model, '../models/best_model')

            print()
        time_elapsed = datetime.now() - since
        print('Complete training in {}'.format(time_elapsed))
        print('Best val Acc: {:.4f}'.format(best_acc))

        return best_model, {'best_acc': best_acc, 'time_elapsed': time_elapsed}

def multicriterion(outputs, labels):
    loss = []
    for i, out in enumerate(outputs):
        loss_current = nn.functional.nll_loss(out, labels[i], )
        loss.append(loss_current)
    return torch.cat(loss).sum()


def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer

#def lr_scheduler(optimizer, divide):
#    if divide:
#        for param_group in optimizer.param_groups:
#            param_group['lr'] = param_group['lr'] / 2
#            print('Set lr to ', param_group['lr'])
#    return optimizer
