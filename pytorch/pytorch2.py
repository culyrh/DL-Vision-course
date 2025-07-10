# 4. DatasetDataLoaderExample
# Dataset 정의
import torch

x_train = torch.Tensor([1, 2, 3, 4, 5, 6]).view(6,1)
y_train = torch.Tensor([3, 4, 5, 6, 7, 8]).view(6,1)



from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.x_train.shape[0]
    


# Dataset 인스턴스 / DataLoader 인스턴스 생성
dataset = CustomDataset(x_train, y_train)

train_loader = DataLoader(dataset=dataset, batch_size=3, shuffle=True)



total_batch = len(train_loader)

print(total_batch)
# 2





# 신경망 모델 구축
from torch import nn

class MyLinearRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(1, 1)
        )

    def forward(self, data):
        prediction = self.linear_stack(data)

        return prediction



# 모델 생성
model = MyLinearRegressionModel()



# 손실함수 및 옵티마이저 설정
loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)



for epoch in range(2):

    for idx, batch_data in enumerate(train_loader):

        x_train_batch, y_train_batch = batch_data

        output_batch = model(x_train_batch)

        print('==============================================')
        print('epoch =', epoch+1, ', batch_idx =', idx+1, ',',
              len(x_train_batch), len(y_train_batch), len(output_batch))
        print('==============================================')

        loss = loss_function(output_batch, y_train_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# ==============================================
# epoch = 1 , batch_idx = 1 , 3 3 3
# ==============================================
# ==============================================
# epoch = 1 , batch_idx = 2 , 3 3 3
# ==============================================
# ==============================================
# epoch = 2 , batch_idx = 1 , 3 3 3
# ==============================================
# ==============================================
# epoch = 2 , batch_idx = 2 , 3 3 3
# ==============================================





# 5. DeepLearningBasicExample
# 데이터 정의
import torch

x_train = torch.Tensor([2, 4, 6, 8, 10,
                        12, 14, 16, 18, 20]).view(10,1)
y_train = torch.Tensor([0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1]).view(10,1)

print(x_train.shape, y_train.shape)
# torch.Size([10, 1]) torch.Size([10, 1])



# 신경망 모델 구축
from torch import nn

class MyDeepLearningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.deeplearning_stack = nn.Sequential(
            nn.Linear(1, 8),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        prediction = self.deeplearning_stack(data)
        return prediction



deeplearning_model = MyDeepLearningModel()

for name, child in deeplearning_model.named_children():
    for param in child.parameters():
        print(name, param)
# deeplearning_stack Parameter containing:
# tensor([[-0.7720],
#         [ 0.9843],
#         [-0.7316],
#         [-0.0175],
#         [-0.8554],
#         [ 0.2618],
#         [ 0.5337],
#         [-0.9485]], requires_grad=True)
# deeplearning_stack Parameter containing:
# tensor([ 0.7829, -0.4415,  0.3658,  0.3090,  0.8152,  0.4430, -0.6894, -0.6758],
#        requires_grad=True)
# deeplearning_stack Parameter containing:
# tensor([[-0.2815, -0.1281, -0.1862,  0.0398, -0.0020,  0.1239,  0.0337,  0.0225]],
#        requires_grad=True)
# deeplearning_stack Parameter containing:
# tensor([0.1685], requires_grad=True)



# 손실함수 및 옵티마이저 설정
loss_function = nn.BCELoss()

optimizer = torch.optim.SGD(deeplearning_model.parameters(), lr=1e-1)



nums_epoch = 5000

for epoch in range(nums_epoch+1):

    outputs = deeplearning_model(x_train)

    loss = loss_function(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch = ', epoch, ' current loss = ', loss.item())
# epoch =  0  current loss =  1.1901192665100098
# epoch =  100  current loss =  0.3970009684562683
# epoch =  200  current loss =  0.31602221727371216
# epoch =  300  current loss =  0.26855844259262085
# epoch =  400  current loss =  0.2366284430027008
# epoch =  500  current loss =  0.21335844695568085
# epoch =  600  current loss =  0.1954825520515442
# epoch =  700  current loss =  0.18122707307338715
# epoch =  800  current loss =  0.16953468322753906
# epoch =  900  current loss =  0.1597316563129425
# epoch =  1000  current loss =  0.1513662040233612
# epoch =  1100  current loss =  0.14412327110767365
# epoch =  1200  current loss =  0.13777436316013336
# epoch =  1300  current loss =  0.132152259349823
# epoch =  1400  current loss =  0.12712793052196503
# epoch =  1500  current loss =  0.1226036325097084
# epoch =  1600  current loss =  0.11850112676620483
# epoch =  1700  current loss =  0.11475837230682373
# epoch =  1800  current loss =  0.11132603883743286
# epoch =  1900  current loss =  0.10816188901662827
# epoch =  2000  current loss =  0.10523291677236557
# epoch =  2100  current loss =  0.10251101106405258
# epoch =  2200  current loss =  0.09997135400772095
# epoch =  2300  current loss =  0.09759366512298584
# epoch =  2400  current loss =  0.09536289423704147
# epoch =  2500  current loss =  0.09326116740703583
# epoch =  2600  current loss =  0.09127220511436462
# epoch =  2700  current loss =  0.08941353857517242
# epoch =  2800  current loss =  0.08764719218015671
# epoch =  2900  current loss =  0.08499775826931
# epoch =  3000  current loss =  0.06942813843488693
# epoch =  3100  current loss =  0.03740139305591583
# epoch =  3200  current loss =  0.07625939697027206
# epoch =  3300  current loss =  0.033237237483263016
# epoch =  3400  current loss =  0.03634051978588104
# epoch =  3500  current loss =  0.030959099531173706
# epoch =  3600  current loss =  0.13787995278835297
# epoch =  3700  current loss =  0.0306173674762249
# epoch =  3800  current loss =  0.02684694156050682
# epoch =  3900  current loss =  0.21516147255897522
# epoch =  4000  current loss =  0.029046589508652687
# epoch =  4100  current loss =  0.025493711233139038
# epoch =  4200  current loss =  0.022684114053845406
# epoch =  4300  current loss =  0.020425420254468918
# epoch =  4400  current loss =  0.028740311041474342
# epoch =  4500  current loss =  0.025084849447011948
# epoch =  4600  current loss =  0.02222464047372341
# epoch =  4700  current loss =  0.019920866936445236
# epoch =  4800  current loss =  0.018024230375885963
# epoch =  4900  current loss =  0.016435537487268448
# epoch =  5000  current loss =  0.015085873194038868



for name, child in deeplearning_model.named_children():
    for param in child.parameters():
        print(name, param)
# deeplearning_stack Parameter containing:
# tensor([[-0.4406],
#         [ 0.2187],
#         [-0.2214],
#         [-0.1463],
#         [-0.4181],
#         [-0.1942],
#         [ 0.3542],
#         [ 0.3152]], requires_grad=True)
# deeplearning_stack Parameter containing:
# tensor([ 4.7622, -2.3638,  2.3934,  1.5811,  4.5188,  2.0986, -3.8284, -3.4062],
#        requires_grad=True)
# deeplearning_stack Parameter containing:
# tensor([[-1.5029,  0.7460, -0.7553, -0.4990, -1.4261, -0.6623,  1.2082,  1.0750]],
#        requires_grad=True)
# deeplearning_stack Parameter containing:
# tensor([-5.5218], requires_grad=True)



# test data를 이용한 예측
deeplearning_model.eval()

test_data = torch.Tensor([0.5, 3.0, 3.5, 11.0, 13.0, 31.0]).view(6,1)

pred = deeplearning_model(test_data)

logical_value = (pred > 0.5).float()

print(pred)
print(logical_value)
# In [42]: print(pred)
# tensor([[1.4259e-14],
#         [8.5235e-12],
#         [3.0614e-11],
#         [6.4965e-03],
#         [5.2113e-01],
#         [1.0000e+00]], grad_fn=<SigmoidBackward0>)
# tensor([[0.],
#         [0.],
#         [0.],
#         [0.],
#         [1.],
#         [1.]])





# 6. MLP_FashionMNIST_Example
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np



train_dataset = datasets.FashionMNIST(root='FashionMNIST_data/', train=True,  # 학습 데이터 50000개
                            transform=transforms.ToTensor(), # 0~255까지의 값을 0~1 사이의 값으로 변환시켜줌
                            download=True)

test_dataset = datasets.FashionMNIST(root='FashionMNIST_data/', train=False,  # 테스트 데이터 10000개
                            transform=transforms.ToTensor(), # 0~255까지의 값을 0~1 사이의 값으로 변환시켜줌
                            download=True)
# 100.0%
# 100.0%
# 100.0%
# 100.0%



print(len(train_dataset))

train_dataset_size = int(len(train_dataset) * 0.85)
validation_dataset_size = int(len(train_dataset) * 0.15)

train_dataset, validation_dataset = random_split(train_dataset, [train_dataset_size, validation_dataset_size])

print(len(train_dataset), len(validation_dataset), len(test_dataset))
# 60000
# 51000 9000 10000



BATCH_SIZE = 32

train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

validation_dataset_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)



class MyDeepLearningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, data):
        data = self.flatten(data)
        data = self.fc1(data)
        data = self.relu(data)
        data = self.dropout(data)
        logits = self.fc2(data)
        return logits



model = MyDeepLearningModel()

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)



def model_train(dataloader, model, loss_function, optimizer):

    model.train()

    train_loss_sum = 0
    train_correct = 0
    train_total = 0

    total_train_batch = len(dataloader)

    for images, labels in dataloader: # images에는 이미지, labels에는 0-9 숫자

        # reshape input image into [batch_size by 784]
        # label is not one-hot encoded
        x_train = images.view(-1, 28 * 28) #처음 크기는 (batch_size, 1, 28, 28) / 이걸 (batch_size, 784)로 변환
        y_train = labels

        outputs = model(x_train)
        loss = loss_function(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

        train_total += y_train.size(0)  # label 열 사이즈 같음
        train_correct += ((torch.argmax(outputs, 1)==y_train)).sum().item() # 예측한 값과 일치한 값의 합

    train_avg_loss = train_loss_sum / total_train_batch
    train_avg_accuracy = 100*train_correct / train_total

    return (train_avg_loss, train_avg_accuracy)



def model_evaluate(dataloader, model, loss_function, optimizer):

    model.eval()

    with torch.no_grad(): #미분하지 않겠다는 것

        val_loss_sum = 0
        val_correct=0
        val_total = 0

        total_val_batch = len(dataloader)

        for images, labels in dataloader: # images에는 이미지, labels에는 0-9 숫자

            # reshape input image into [batch_size by 784]
            # label is not one-hot encoded
            x_val = images.view(-1, 28 * 28) #처음 크기는 (batch_size, 1, 28, 28) / 이걸 (batch_size, 784)로 변환
            y_val = labels

            outputs = model(x_val)
            loss = loss_function(outputs, y_val)

            val_loss_sum += loss.item()

            val_total += y_val.size(0)  # label 열 사이즈 같음
            val_correct += ((torch.argmax(outputs, 1)==y_val)).sum().item() # 예측한 값과 일치한 값의 합

        val_avg_loss = val_loss_sum / total_val_batch
        val_avg_accuracy = 100*val_correct / val_total

    return (val_avg_loss, val_avg_accuracy)



def model_test(dataloader, model):

    model.eval()

    with torch.no_grad(): #test set으로 데이터를 다룰 때에는 gradient를 주면 안된다.

        test_loss_sum = 0
        test_correct=0
        test_total = 0

        total_test_batch = len(dataloader)

        for images, labels in dataloader: # images에는 이미지, labels에는 0-9 숫자

            # reshape input image into [batch_size by 784]
            # label is not one-hot encoded
            x_test = images.view(-1, 28 * 28) #처음 크기는 (batch_size, 1, 28, 28) / 이걸 (batch_size, 784)로 변환
            y_test = labels

            outputs = model(x_test)
            loss = loss_function(outputs, y_test)

            test_loss_sum += loss.item()

            test_total += y_test.size(0)  # label 열 사이즈 같음
            test_correct += ((torch.argmax(outputs, 1)==y_test)).sum().item() # 예측한 값과 일치한 값의 합

        test_avg_loss = test_loss_sum / total_test_batch
        test_avg_accuracy = 100*test_correct / test_total

        print('accuracy:', test_avg_accuracy)
        print('loss:', test_avg_loss)



from datetime import datetime

train_loss_list = []
train_accuracy_list = []

val_loss_list = []
val_accuracy_list = []

start_time = datetime.now()

EPOCHS = 20

for epoch in range(EPOCHS):

    #==============  model train  ================
    train_avg_loss, train_avg_accuracy = model_train(train_dataset_loader, model, loss_function, optimizer)  # training

    train_loss_list.append(train_avg_loss)
    train_accuracy_list.append(train_avg_accuracy)
    #=============================================

    #============  model evaluation  ==============
    val_avg_loss, val_avg_accuracy = model_evaluate(validation_dataset_loader, model, loss_function, optimizer)  # evaluation

    val_loss_list.append(val_avg_loss)
    val_accuracy_list.append(val_avg_accuracy)
    #============  model evaluation  ==============

    print('epoch:', '%02d' % (epoch + 1),
          'train loss =', '{:.4f}'.format(train_avg_loss), 'train accuracy =', '{:.4f}'.format(train_avg_accuracy),
          'validation loss =', '{:.4f}'.format(val_avg_loss), 'validation accuracy =', '{:.4f}'.format(val_avg_accuracy))

end_time = datetime.now()

print('elapsed time => ', end_time-start_time)
# epoch: 01 train loss = 0.9889 train accuracy = 67.4843 validation loss = 0.6668 validation accuracy = 77.8889
# epoch: 02 train loss = 0.6191 train accuracy = 78.7627 validation loss = 0.5556 validation accuracy = 81.8333
# epoch: 03 train loss = 0.5473 train accuracy = 81.1431 validation loss = 0.5087 validation accuracy = 83.2444
# epoch: 04 train loss = 0.5053 train accuracy = 82.5980 validation loss = 0.4825 validation accuracy = 83.6222
# epoch: 05 train loss = 0.4788 train accuracy = 83.3961 validation loss = 0.4573 validation accuracy = 84.2889
# epoch: 06 train loss = 0.4568 train accuracy = 84.1745 validation loss = 0.4540 validation accuracy = 84.3222
# epoch: 07 train loss = 0.4406 train accuracy = 84.4882 validation loss = 0.4315 validation accuracy = 85.2222
# epoch: 08 train loss = 0.4270 train accuracy = 85.0353 validation loss = 0.4241 validation accuracy = 85.4667
# epoch: 09 train loss = 0.4151 train accuracy = 85.3216 validation loss = 0.4131 validation accuracy = 85.9889
# epoch: 10 train loss = 0.4042 train accuracy = 85.8431 validation loss = 0.4034 validation accuracy = 86.2000
# epoch: 11 train loss = 0.3958 train accuracy = 86.0392 validation loss = 0.3968 validation accuracy = 86.4000
# epoch: 12 train loss = 0.3876 train accuracy = 86.3608 validation loss = 0.3910 validation accuracy = 86.6111
# epoch: 13 train loss = 0.3814 train accuracy = 86.5980 validation loss = 0.3879 validation accuracy = 86.4000
# epoch: 14 train loss = 0.3746 train accuracy = 86.7196 validation loss = 0.3778 validation accuracy = 87.1444
# epoch: 15 train loss = 0.3683 train accuracy = 86.9412 validation loss = 0.3803 validation accuracy = 86.7667
# epoch: 16 train loss = 0.3640 train accuracy = 87.0569 validation loss = 0.3721 validation accuracy = 87.0778
# epoch: 17 train loss = 0.3590 train accuracy = 87.3314 validation loss = 0.3675 validation accuracy = 87.2111
# epoch: 18 train loss = 0.3541 train accuracy = 87.4784 validation loss = 0.3640 validation accuracy = 87.1333
# epoch: 19 train loss = 0.3501 train accuracy = 87.5176 validation loss = 0.3705 validation accuracy = 86.8889
# epoch: 20 train loss = 0.3450 train accuracy = 87.7961 validation loss = 0.3635 validation accuracy = 87.2667
# elapsed time =>  0:06:23.605847



# test dataset 으로 정확도 및 오차 테스트
model_test(test_dataset_loader, model)
# accuracy: 86.22
# loss: 0.3805399534944147



import matplotlib.pyplot as plt

plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(train_loss_list, label='train loss')
plt.plot(val_loss_list, label='validation loss')

plt.legend()

plt.show()
# ->



import matplotlib.pyplot as plt

plt.title('Accuracy Trend')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(train_accuracy_list, label='train accuracy')
plt.plot(val_accuracy_list, label='validation accuracy')

plt.legend()

plt.show()
# ->



plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.plot(train_loss_list, label='train')
plt.plot(val_loss_list, label='validation')
plt.legend()

plt.subplot(1,2,2)
plt.title('Accuracy Trend')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()
plt.plot(train_accuracy_list, label='train')
plt.plot(val_accuracy_list, label='validation')
plt.legend()

plt.show()
# ->