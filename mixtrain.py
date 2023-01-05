import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from torch.utils.data import TensorDataset
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from conf import settings
from sklearn.manifold import TSNE


def get_training_dataloader(t,ratio, batch_size=512, num_workers=2, shuffle=True):
    transform_train = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5) )
    ])
    L = t
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                      transform=transform_train)
    mnist_training = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    mnist_training.data, mnist_training.targets = mnist_training.data[:5000], mnist_training.targets[:5000]
    mnist_training.data = np.expand_dims(mnist_training.data, axis=-1)
    mnist_training.data = np.pad(mnist_training.data, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    mnist_training.data = np.concatenate((mnist_training.data, mnist_training.data, mnist_training.data), axis=3)

    data, target = cifar100_training.data, cifar100_training.targets
    indexes = []
    for i, j in enumerate(cifar100_training.targets):
        if j in L:
            continue
        else:
            indexes.append(i)
    data = np.delete(data, indexes, axis=0)
    target = [L.index(i) + int(ratio * 10) for i in target if i in L]

    cifar100_training.data = data
    cifar100_training.targets = target

    data, target = mnist_training.data, mnist_training.targets
    indexes = []
    for i, j in enumerate(mnist_training.targets):
        if int(j) in range(int(ratio * 10)):
            continue
        else:
            indexes.append(i)
    data = np.delete(data, indexes, axis=0)
    target = [i for i in target if int(i) in range(int(ratio * 10))]

    cifar100_training.data = np.concatenate((cifar100_training.data, data), axis=0)
    cifar100_training.targets = np.array(cifar100_training.targets + target)

    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(t,ratio, batch_size=512, num_workers=2, shuffle=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    L = t
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)


    data, target = cifar100_test.data, cifar100_test.targets
    indexes = []
    for i, j in enumerate(cifar100_test.targets):
        if j in L:
            continue
        else:
            indexes.append(i)
    data = np.delete(data, indexes, axis=0)
    target = [L.index(i) + int(ratio * 10) for i in target if i in L]

    cifar100_test.data = data
    cifar100_test.targets = target


    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def train(epoch):
    net.train()
    loss_function = nn.CrossEntropyLoss()
    for i in range(epoch):
        start = time.time()
        for batch_index, (images, labels) in enumerate(cifar100_training_loader):
            labels = labels.to(device,dtype=torch.int64)
            images = images.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=i,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(cifar100_training_loader.dataset)
            ))
            finish = time.time()
            print('epoch {} training time consumed: {:.2f}s'.format(i, finish - start))
        scheduler.step()



def get_test_onlycifar(t,ratio, batch_size=512, num_workers=2, shuffle=True,):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    L=t
    cifar100_test = torchvision.datasets.CIFAR100(root='~/data', train=False, download=True, transform=transform_test)

    data, target = cifar100_test.data, cifar100_test.targets
    indexes = []
    if L==[]:
        data=np.array([])
        target=[]
    else:
        for i, j in enumerate(cifar100_test.targets):
            if j in L:
                continue
            else:
                indexes.append(i)
        data = np.delete(data, indexes, axis=0)
        target = [L.index(i) + int(ratio * 10) for i in target if i in L]

    cifar100_test.data = data
    cifar100_test.targets = target
    test_loader=DataLoader(cifar100_test,shuffle=shuffle,num_workers=num_workers,batch_size=batch_size)
    return test_loader


def test():
    net.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    model=TSNE(learning_rate=100)

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
            image = image.to(device)
            label = label.to(device,dtype=torch.int64)
            output = net(image)


            _, pred = output.topk(5, 1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            # compute top 5
            correct_5 += correct[:, :5].sum()
            # compute top1
            correct_1 += correct[:, :1].sum()

    print("Top 1 acc: ", correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 acc: ", correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))



def test_cifar():
    net.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    model=TSNE(learning_rate=100)

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(only_cifar_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
            image = image.cuda()
            label = label.cuda()
            output = net(image)


            _, pred = output.topk(5, 1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            # compute top 5
            correct_5 += correct[:, :5].sum()
            # compute top1
            correct_1 += correct[:, :1].sum()


    print("Top 1 acc: ",  correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 acc: ", correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))







if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ratio', type=float, default=0.5, help='#ofsvhn/#ofcifar100')
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-t', type=str, default='s', help='g:gaussian selection,t:imbalanced token')
    parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate')
    args = parser.parse_args()
    L=[i for i in range(10,100)]
    num_cifar=10-int(args.ratio*10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.t=='s':
        L = [12, 22, 32, 42, 52, 62, 72, 82, 92, 99]

        print('selected token:', sorted(L))
    else:
        L=[34,36,50,63,64,65,66,74,75,80]

        print('selected token:', sorted(L))
    net=torchvision.models.resnet18(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, 20)
    net=net.to(device)

    cifar100_training_loader = get_training_dataloader(
        L,
        args.ratio,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
    )
    cifar100_test_loader = get_test_dataloader(
        L,
        args.ratio,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(cifar100_training_loader), eta_min=0,
                                                           last_epoch=-1)
    train(20)
    test()


