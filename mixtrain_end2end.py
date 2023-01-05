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


def get_training_dataloader(t,ratio, batch_size=256, num_workers=2, shuffle=True):
    transform_train = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    L=t
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    mnist_training = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform_train)
    mnist_training.data,mnist_training.targets=mnist_training.data[:5000],mnist_training.targets[:5000]
    mnist_training.data=np.expand_dims(mnist_training.data,axis=-1)
    mnist_training.data=np.pad(mnist_training.data,((0,0),(2,2),(2,2),(0,0)),'constant')
    mnist_training.data=np.concatenate((mnist_training.data,mnist_training.data,mnist_training.data),axis=3)

    data,target = cifar100_training.data,cifar100_training.targets
    indexes= []
    for i,j in enumerate(cifar100_training.targets):
        if j in L:
            continue
        else:
            indexes.append(i)
    data=np.delete(data,indexes,axis=0)
    target=[L.index(i)+int(ratio*10) for i in target if i in L]

    cifar100_training.data=data
    cifar100_training.targets=target

    data,target= mnist_training.data,mnist_training.targets
    indexes = []
    for i, j in enumerate(mnist_training.targets):
        if int(j) in range(int(ratio*10)):
            continue
        else:
            indexes.append(i)
    data = np.delete(data, indexes, axis=0)
    target = [i for i in target if int(i) in range(int(ratio*10))]

    cifar100_training.data=np.concatenate((cifar100_training.data,data),axis=0)
    cifar100_training.targets=np.array(cifar100_training.targets+target)

    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(t,ratio, batch_size=512, num_workers=2, shuffle=True):
    transform_test = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    L=t
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
            labels = labels.to(device=device, dtype=torch.int64)
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




def test(token):
    net.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    model=TSNE(learning_rate=100)

    with torch.no_grad():
        count=0
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            image = image.cuda()
            label = label.to(device)
            output = net(image)


            _, pred = output.topk(5, 1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            # compute top 5
            correct_5 += correct[:, :5].sum()
            # compute top1
            correct_1 += correct[:, :1].sum()



    #plt.show()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
    if token == Randomtoken:
        y_R.append(correct_1 / len(cifar100_test_loader.dataset))
    else:
        y_S.append(correct_1 / len(cifar100_test_loader.dataset))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=3e-4, help='initial learning rate')
    args = parser.parse_args()
    L=[i for i in range(10,100)]
    x=np.array([0,0.1,0.2,0.3,0.4,0.5])
    y_R=[];y_S=[]


    Randomtoken=[12,22,32,42,52,61,69,82,92,99]

    Similartoken=[34,36,50,63,64,65,66,74,75,80]

    for token in [Randomtoken,Similartoken]:
        for ratio in x:
            print('############')
            print('ratio=',ratio)
            num_cifar = 10-int(10 * ratio)
            print(num_cifar)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            net = torchvision.models.resnet18(pretrained=True)
            net.fc = nn.Linear(net.fc.in_features, 20)

            '''for name,param in net.named_parameters():
                if name not in ['fc.weight','fc.bias']:
                    param.requires_grad=False
            net.fc.weight.data.normal_(mean=0.0,std=0.01)
            net.fc.bias.data.zero_()
            parameters=list(filter(lambda p: p.requires_grad,net.parameters()))'''


            net = net.to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,weight_decay=1e-4)
            cifar100_training_loader = get_training_dataloader(
                    token,
                    ratio,
                    num_workers=4,
                    batch_size=args.b,
                    shuffle=True,

            )

            cifar100_test_loader = get_test_dataloader(
                    token,
                    ratio,
                    num_workers=4,
                    batch_size=args.b,
                    shuffle=True,

            )

            train(20)
            test(token)


    plt.plot(x,np.array(y_R),label='Simple(SL)')
    plt.plot(x,np.array(y_S),label='Hard(SL)')
    y_R=[0.6384,0.5688,0.59  ,0.608 ,0.6004,0.5978]
    y_S=[0.4203,0.3667,0.3746,0.3924,0.3634,0.37]
    #y_R=[0.289,0.2596,0.2886,0.2416,0.291,0.3008]
    #y_S=[0.1968,0.176,0.2048,0.1696]

    plt.plot(x,y_R,label='Simple(SSL)')
    plt.plot(x,y_S,label='Hard(SSL)')


    plt.xlabel('ratio')
    plt.ylabel('accuracy')


    plt.legend()
    plt.show()