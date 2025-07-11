# GPU 사용을 위해 파이썬을 사용하지 않고
# colab에서 CUDA 12.4 버전으로 빌드된 GPU를 사용하여 진행한다

# AlexNet, VGG, ResNet 모델 구현


##############################
# 1. AlexNet
##############################

# AlexNet 구현
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch

import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
# DataSet: hymenoptera_data(#classes : 2)
# hymenoptera_data



ddir = '{개별경로설정}'

batch_size = 64
num_workers = 4

data_transformers = {
    'train': transforms.Compose(
        [
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.490, 0.449, 0.411], [0.231, 0.221, 0.230])
        ]
    ),
    'val': transforms.Compose(
        [
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.490, 0.449, 0.411],[0.231, 0.221, 0.230])
        ]
    )
}

img_data = {
    k: datasets.ImageFolder(os.path.join(ddir, k), data_transformers[k])
    for k in ['train', 'val']
}
dloaders = {
    k: torch.utils.data.DataLoader(
        img_data[k], batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    for k in ['train', 'val']
}
dset_sizes = {x: len(img_data[x]) for x in ['train', 'val']}
classes = img_data['train'].classes

dvc = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# /usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:560: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
# cpuset_checked))



# 모델구현 Alexnet
class AlexNet(nn.Module):
  def __init__(self, number_of_classes):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3, 96, kernel_size=11, stride=4),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2)
        )

    self.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 5 * 5, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, number_of_classes),
        )

  def forward(self, inp):
    op = self.features(inp)

    op = op.view(op.size(0), -1)
    op = self.classifier(op)
    return op
  


  # 학습
  def train(model, loss_func, optimizer, epochs=10):
   start = time.time()

   accuracy = 0.0

   for e in range(epochs):
        print(f'Epoch number {e}/{epochs - 1}')
        print('=' * 20)

        for dset in ['train', 'val']:
            if dset == 'train':
                model.train()
            else:
                model.eval()

            loss = 0.0
            successes = 0

            for imgs, tgts in dloaders[dset]:
                imgs = imgs.to(dvc)
                tgts = tgts.to(dvc)
                optimizer.zero_grad()

                with torch.set_grad_enabled(dset == 'train'):
                    ops = model(imgs)
                    _, preds = torch.max(ops, 1)
                    loss_curr = loss_func(ops, tgts)

                    if dset == 'train':
                        loss_curr.backward()
                        optimizer.step()

                loss += loss_curr.item() * imgs.size(0)
                successes += torch.sum(preds == tgts.data)

            loss_epoch = loss / dset_sizes[dset]
            accuracy_epoch = successes.double() / dset_sizes[dset]

            print(f'{dset} loss in this epoch: {loss_epoch}, accuracy in this epoch: {accuracy_epoch}')
            if dset == 'val' and accuracy_epoch > accuracy:
                accuracy = accuracy_epoch

   time_delta = time.time() - start
   print(f'Training finished in {time_delta // 60}mins {time_delta % 60}secs')
   print(f'Best validation set accuracy: {accuracy}')


   return model
  


model = AlexNet(2)
if torch.cuda.is_available() :
  model = model.cuda()
print(model)
# AlexNet(
# (features): Sequential(
# (0): Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4))
# (1): ReLU()
# (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
# (3): Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
# (4): ReLU()
# (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
# (6): Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (7): ReLU()
# (8): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (9): ReLU()
# (10): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (11): ReLU()
# (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
# )
# (classifier): Sequential(
# (0): Dropout(p=0.5, inplace=False)
# (1): Linear(in_features=6400, out_features=4096, bias=True)
# (2): ReLU()
# (3): Dropout(p=0.5, inplace=False)
# (4): Linear(in_features=4096, out_features=4096, bias=True)
# (5): ReLU()
# (6): Linear(in_features=4096, out_features=2, bias=True) 
# )
# )



loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
pretrained_model = train(model, loss_func, optimizer, epochs=5)
# Epoch number 0/4
# ====================
# /usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:560: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
#   cpuset_checked))
# train loss in this epoch: 0.6991265142550234, accuracy in this epoch: 0.4713114754098361
# val loss in this epoch: 0.6909760100389618, accuracy in this epoch: 0.542483660130719
# Epoch number 1/4
# ====================
# train loss in this epoch: 0.6927160024642944, accuracy in this epoch: 0.5122950819672132
# val loss in this epoch: 0.6985304943876329, accuracy in this epoch: 0.4575163398692811
# Epoch number 2/4
# ====================
# train loss in this epoch: 0.6903637461974973, accuracy in this epoch: 0.5286885245901639
# val loss in this epoch: 0.6916465159335168, accuracy in this epoch: 0.49673202614379086
# Epoch number 3/4
# ====================
# train loss in this epoch: 0.6810294037959614, accuracy in this epoch: 0.5819672131147541
# val loss in this epoch: 0.669171616922017, accuracy in this epoch: 0.5882352941176471
# Epoch number 4/4
# ====================
# train loss in this epoch: 0.664376226604962, accuracy in this epoch: 0.5778688524590164
# val loss in this epoch: 0.664067312782886, accuracy in this epoch: 0.6013071895424836
# Training finished in 0.0mins 24.318676948547363secs
# Best validation set accuracy: 0.6013071895424836



# Test
def test(model=pretrained_model):

  correct_pred = {classname: 0 for classname in classes}
  total_pred = {classname: 0 for classname in classes}


  with torch.no_grad():
      for images, labels in dloaders['val']:


          images = images.to(dvc)
          labels = labels.to(dvc)

          outputs = model(images)
          _, predictions = torch.max(outputs, 1)

          for label, prediction in zip(labels, predictions):
              if label == prediction:
                  correct_pred[classes[label]] += 1
              total_pred[classes[label]] += 1



  for classname, correct_count in correct_pred.items():
      accuracy = 100 * float(correct_count) / total_pred[classname]
      print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
test(pretrained_model)
# /usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:560: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
# cpuset_checked))
# Accuracy for class: ants is 34.3 %
# Accuracy for class: bees is 81.9 %



# image 확인
def imageshow(img, text=None):
  img = img.numpy().transpose((1,2,0))
  avg = np.array([0.490, 0.449, 0.411])
  stddev = np.array([0.231, 0.221, 0.230])
  img = stddev * img + avg
  img = np.clip(img, 0, 1)
  plt.imshow(img)
  if text is not None:
    plt.title(text)



def visualize_predictions(pretrained_model, max_num_imgs=4):
    torch.manual_seed(1)
    was_model_training = pretrained_model.training
    pretrained_model.eval()
    imgs_counter = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (imgs, tgts) in enumerate(dloaders['val']):
            imgs = imgs.to(dvc)
            tgts = tgts.to(dvc)
            ops = pretrained_model(imgs)
            _, preds = torch.max(ops, 1)

            for j in range(imgs.size()[0]):
                imgs_counter += 1
                ax = plt.subplot(max_num_imgs//2, 2, imgs_counter)
                ax.axis('off')
                ax.set_title(f'pred: {classes[preds[j]]} || target: {classes[tgts[j]]}')
                imageshow(imgs.cpu().data[j])

                if imgs_counter == max_num_imgs:
                    pretrained_model.train(mode=was_model_training)
                    return
        pretrained_model.train(mode=was_model_training)



visualize_predictions(pretrained_model)



##############################
# 2. VGG
##############################

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch

import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms



ddir = '{개별경로설정}'

batch_size = 64
num_workers = 2

data_transformers = {
    'train': transforms.Compose(
        [
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.490, 0.449, 0.411], [0.231, 0.221, 0.230])
        ]
    ),
    'val': transforms.Compose(
        [
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.490, 0.449, 0.411],[0.231, 0.221, 0.230])
        ]
    )
}

img_data = {
    k: datasets.ImageFolder(os.path.join(ddir, k), data_transformers[k])
    for k in ['train', 'val']
}
dloaders = {
    k: torch.utils.data.DataLoader(
        img_data[k], batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    for k in ['train', 'val']
}
dset_sizes = {x: len(img_data[x]) for x in ['train', 'val']}
classes = img_data['train'].classes

dvc = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



import torch
import torch.nn as nn

cfg = {
    'vgg16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'vgg19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]


        layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def vgg16(num=100):
    return VGG(make_layers(cfg['vgg16']),num_classes=num)

def vgg19(num=100):
    return VGG(make_layers(cfg['vgg19']),num_classes=num)



def train(model, loss_func, optimizer, epochs=10):
   start = time.time()

   accuracy = 0.0

   for e in range(epochs):
        print(f'Epoch number {e}/{epochs - 1}')
        print('=' * 20)

        for dset in ['train', 'val']:
            if dset == 'train':
                model.train()
            else:
                model.eval()

            loss = 0.0
            successes = 0

            for imgs, tgts in dloaders[dset]:
                imgs = imgs.to(dvc)
                tgts = tgts.to(dvc)
                optimizer.zero_grad()

                with torch.set_grad_enabled(dset == 'train'):
                    ops = model(imgs)
                    _, preds = torch.max(ops, 1)
                    loss_curr = loss_func(ops, tgts)

                    if dset == 'train':
                        loss_curr.backward()
                        optimizer.step()

                loss += loss_curr.item() * imgs.size(0)
                successes += torch.sum(preds == tgts.data)

            loss_epoch = loss / dset_sizes[dset]
            accuracy_epoch = successes.double() / dset_sizes[dset]

            print(f'{dset} loss in this epoch: {loss_epoch}, accuracy in this epoch: {accuracy_epoch}')
            if dset == 'val' and accuracy_epoch > accuracy:
                accuracy = accuracy_epoch

   time_delta = time.time() - start
   print(f'Training finished in {time_delta // 60}mins {time_delta % 60}secs')
   print(f'Best validation set accuracy: {accuracy}')


   return model



model = vgg19(2)
if torch.cuda.is_available():
  model = model.cuda()
print(model)
# VGG(
# (features): Sequential(
# (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (2): ReLU(inplace=True)
# (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (5): ReLU(inplace=True)
# (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (9): ReLU(inplace=True)
# (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (12): ReLU(inplace=True)
# (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (16): ReLU(inplace=True)
# (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (19): ReLU(inplace=True)
# (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (22): ReLU(inplace=True)
# (23): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (24): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (25): ReLU(inplace=True)
# (26): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# (27): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (29): ReLU(inplace=True)
# (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (32): ReLU(inplace=True)
# (33): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (34): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (35): ReLU(inplace=True)
# (36): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (37): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (38): ReLU(inplace=True)
# (39): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (42): ReLU(inplace=True)
# (43): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (44): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (45): ReLU(inplace=True)
# (46): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (47): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (48): ReLU(inplace=True)
# (49): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (50): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (51): ReLU(inplace=True)
# (52): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# )
# (classifier): Sequential(
# (0): Linear(in_features=25088, out_features=4096, bias=True)
# (1): ReLU(inplace=True)
# (2): Dropout(p=0.5, inplace=False)
# (3): Linear(in_features=4096, out_features=4096, bias=True)
# (4): ReLU(inplace=True)
# (5): Dropout(p=0.5, inplace=False)
# (6): Linear(in_features=4096, out_features=2, bias=True)
# )
# )



loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
pretrained_model = train(model, loss_func, optimizer, epochs=5)
# Epoch number 0/4
# ====================
# train loss in this epoch: 2.7115575563712198, accuracy in this epoch: 0.4836065573770492
# val loss in this epoch: 0.6913309167413151, accuracy in this epoch: 0.542483660130719
# Epoch number 1/4
# ====================
# train loss in this epoch: 1.081803807469665, accuracy in this epoch: 0.5040983606557378
# val loss in this epoch: 0.7028964317702, accuracy in this epoch: 0.4575163398692811
# Epoch number 2/4
# ====================
# train loss in this epoch: 0.8111628352618608, accuracy in this epoch: 0.5245901639344263
# val loss in this epoch: 0.7531087001164755, accuracy in this epoch: 0.4575163398692811
# Epoch number 3/4
# ====================
# train loss in this epoch: 0.6686882542782142, accuracy in this epoch: 0.569672131147541
# val loss in this epoch: 0.6915509318993762, accuracy in this epoch: 0.542483660130719
# Epoch number 4/4
# ====================
# train loss in this epoch: 0.7621770843130643, accuracy in this epoch: 0.5
# val loss in this epoch: 0.6984418381273357, accuracy in this epoch: 0.4575163398692811
# Training finished in 1.0mins 22.999370098114014secs
# Best validation set accuracy: 0.542483660130719



def test(model=pretrained_model):

  correct_pred = {classname: 0 for classname in classes}
  total_pred = {classname: 0 for classname in classes}


  with torch.no_grad():
      for images, labels in dloaders['val']:


          images = images.to(dvc)
          labels = labels.to(dvc)

          outputs = model(images)
          _, predictions = torch.max(outputs, 1)

          for label, prediction in zip(labels, predictions):
              if label == prediction:
                  correct_pred[classes[label]] += 1
              total_pred[classes[label]] += 1



  for classname, correct_count in correct_pred.items():
      accuracy = 100 * float(correct_count) / total_pred[classname]
      print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')



test(pretrained_model)
# Accuracy for class: ants is 100.0 %
# Accuracy for class: bees is 0.0 %



def imageshow(img, text=None):
  img = img.numpy().transpose((1,2,0))
  avg = np.array([0.490, 0.449, 0.411])
  stddev = np.array([0.231, 0.221, 0.230])
  img = stddev * img + avg
  img = np.clip(img, 0, 1)
  plt.imshow(img)
  if text is not None:
    plt.title(text)



def visualize_predictions(pretrained_model, max_num_imgs=4):
    torch.manual_seed(1)
    was_model_training = pretrained_model.training
    pretrained_model.eval()
    imgs_counter = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (imgs, tgts) in enumerate(dloaders['val']):
            imgs = imgs.to(dvc)
            tgts = tgts.to(dvc)
            ops = pretrained_model(imgs)
            _, preds = torch.max(ops, 1)

            for j in range(imgs.size()[0]):
                imgs_counter += 1
                ax = plt.subplot(max_num_imgs//2, 2, imgs_counter)
                ax.axis('off')
                ax.set_title(f'pred: {classes[preds[j]]} || target: {classes[tgts[j]]}')
                imageshow(imgs.cpu().data[j])

                if imgs_counter == max_num_imgs:
                    pretrained_model.train(mode=was_model_training)
                    return
        pretrained_model.train(mode=was_model_training)



visualize_predictions(pretrained_model)





##############################
# 3-1. ResNet18_CIFAR10_Train
##############################

# ResNet18 모델 정의 및 인스턴스 초기화
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os

# ResNet18을 위해 최대한 간단히 수정한 BasicBlock 클래스 정의
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        # 3x3 필터를 사용 (너비와 높이를 줄일 때는 stride 값 조절)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) # 배치 정규화(batch normalization)

        # 3x3 필터를 사용 (패딩을 1만큼 주기 때문에 너비와 높이가 동일)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) # 배치 정규화(batch normalization)

        self.shortcut = nn.Sequential() # identity인 경우
        if stride != 1: # stride가 1이 아니라면, Identity mapping이 아닌 경우
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # (핵심) skip connection
        out = F.relu(out)
        return out


# ResNet 클래스 정의
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # 64개의 3x3 필터(filter)를 사용
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes # 다음 레이어를 위해 채널 수 변경
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ResNet18 함수 정의
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])



# 데이터셋(Dataset) 다운로드 및 불러오기
import torchvision
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
# Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
# 170500096/? [00:20<00:00, 100483168.85it/s]
# Extracting ./data/cifar-10-python.tar.gz to ./data
# Files already downloaded and verified



# 환경 설정 및 학습(Training) 함수 정의
device = 'cuda'

net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

learning_rate = 0.1
file_name = 'resnet18_cifar10.pt'

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)


def train(epoch):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        benign_outputs = net(inputs)
        loss = criterion(benign_outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = benign_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current benign train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current benign train loss:', loss.item())

    print('\nTotal benign train accuarcy:', 100. * correct / total)
    print('Total benign train loss:', train_loss)


def test(epoch):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        outputs = net(inputs)
        loss += criterion(outputs, targets).item()

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

    print('\nTest accuarcy:', 100. * correct / total)
    print('Test average loss:', loss / total)

    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name)
    print('Model Saved!')


def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# 학습(Training) 진행
# 대략 20번의 epoch 이후에도 85% 가량의 test accuracy를 얻을 수 있다

# for epoch in range(0, 200):
for epoch in range(0, 20):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    test(epoch)
# [ Train epoch: 0 ]
# 
# Current batch: 0
# Current benign train accuracy: 0.078125
# Current benign train loss: 2.342548370361328
# 
# Current batch: 100
# Current benign train accuracy: 0.234375
# Current benign train loss: 1.9885613918304443
# 
# Current batch: 200
# Current benign train accuracy: 0.3984375
# Current benign train loss: 1.592935562133789
# 
# Current batch: 300
# Current benign train accuracy: 0.375
# Current benign train loss: 1.6573998928070068
# 
# Total benign train accuarcy: 32.606
# Total benign train loss: 728.0853297710419
# 
# [ Test epoch: 0 ]
# 
# Test accuarcy: 32.37
# Test average loss: 0.019251463556289674
# Model Saved!
# (... 출력 생략)
# [ Train epoch: 19 ]
# 
# Current batch: 0
# Current benign train accuracy: 0.8984375
# Current benign train loss: 0.36447006464004517
#
# Current batch: 100
# Current benign train accuracy: 0.8828125
# Current benign train loss: 0.3062523901462555
# 
# Current batch: 200
# Current benign train accuracy: 0.8828125
# Current benign train loss: 0.3175150454044342
# 
# Current batch: 300
# Current benign train accuracy: 0.9296875
# Current benign train loss: 0.19365999102592468
# 
# Total benign train accuarcy: 90.082
# Total benign train loss: 111.900165989995
# 
# [ Test epoch: 19 ]
# 
# Test accuarcy: 86.42
# Test average loss: 0.004216953121125698
# Model Saved!




##############################
# 3-2. ResNet
##############################

# 생략