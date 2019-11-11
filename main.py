from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv

from models.resnet import *
from utils import *


from data.cifar import CIFAR10, CIFAR100
from TruncatedLoss import TruncatedLoss
parser = argparse.ArgumentParser(
    description='PyTorch TruncatedLoss')

parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--sess', default='default', type=str, help='session id')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--decay', default=1e-4, type=float,
                    help='weight decay (default=1e-4)')
parser.add_argument('--lr', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--batch-size', '-b', default=128,
                    type=int, help='mini-batch size (default: 128)')
parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start_prune', default=40, type=int,
                    help='number of total epochs to run')
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='pairflip')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--gamma', type = float, default = 0.1)
parser.add_argument('--schedule', nargs='+', type=int)

best_acc = 0
args = parser.parse_args()
def main():
    
    use_cuda = torch.cuda.is_available()
    global best_acc 
 
    # load dataset
        
    if args.dataset=='cifar10':
        num_classes=10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ])

        train_dataset = CIFAR10(root='./data/',
                                    download=True,  
                                    train=True, 
                                    transform=transform_train,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                               )
        
        test_dataset = CIFAR10(root='./data/',
                                    download=True,  
                                    train=False, 
                                    transform=transform_test,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                              )

    if args.dataset=='cifar100':
        num_classes=100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
        ])
        train_dataset = CIFAR100(root='./data/',
                                    download=True,  
                                    train=True, 
                                    transform=transform_train,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                                )
        
        test_dataset = CIFAR100(root='./data/',
                                    download=True,  
                                    train=False, 
                                    transform=transform_test,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                                )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=2)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # Model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7.' + args.sess)
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    else:
        print('==> Building model.. (Default : ResNet34)')
        start_epoch = 0
        net = ResNet34(num_classes)

    result_folder = './results/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    logname = result_folder + net.__class__.__name__ + \
        '_' + args.sess + '.csv'

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.benchmark = True
        print('Using CUDA..')


    criterion = TruncatedLoss(trainset_size=len(train_dataset)).cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.schedule, gamma=args.gamma)

    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(
                ['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

    for epoch in range(start_epoch, args.epochs):
        
        train_loss, train_acc = train(epoch, trainloader, net, criterion, optimizer)
        test_loss, test_acc = test(epoch, testloader, net, criterion)

        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
        scheduler.step()

# Training
def train(epoch, trainloader, net, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if (epoch+1) >= args.start_prune and (epoch+1) % 10 == 0:
        checkpoint = torch.load('./checkpoint/ckpt.t7.' + args.sess)
        net = checkpoint['net']
        net.eval()
        for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            criterion.update_weight(outputs, targets, indexes)
        now = torch.load('./checkpoint/current_net')
        net = now['current_net']
        net.train()
    
    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets, indexes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
             
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


    return (train_loss / batch_idx, 100. * correct / total)


def test(epoch, testloader, net, criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets, indexes)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        checkpoint(acc, epoch, net)
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    state = {
        'current_net': net,
    }
    torch.save(state, './checkpoint/current_net')
    return (test_loss / batch_idx, 100. * correct / total)


def checkpoint(acc, epoch, net):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7.' +
               args.sess)




if __name__ == '__main__':
    main()