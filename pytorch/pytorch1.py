# pip install torch torchvision





# 1. Tensor_SimpleRegression
# Tensor 만들기
import torch
import numpy as np

list_data = [ [10, 20], [30, 40] ]

tensor1 = torch.Tensor(list_data)

print(tensor1)
print(f"tensor type: {type(tensor1)}, tensor shape: {tensor1.shape}")
print(f"tensor dtype: {tensor1.dtype}, tensor device: {tensor1.device}")
# tensor([[10., 20.],
#        [30., 40.]])
# tensor type: <class 'torch.Tensor'>, tensor shape: torch.Size([2, 2])
# tensor dtype: torch.float32, tensor device: cpu





if torch.cuda.is_available():
    tensor1 = tensor1.to("cuda")

print(f"tensor type: {type(tensor1)}, tensor shape: {tensor1.shape}")
print(f"tensor dtype: {tensor1.dtype}, tensor device: {tensor1.device}")
# tensor type: <class 'torch.Tensor'>, tensor shape: torch.Size([2, 2])
# tensor dtype: torch.float32, tensor device: cpu





numpy_data = np.array(list_data)

tensor2 = torch.Tensor(numpy_data)

print(tensor2)
print(f"tensor type: {type(tensor2)}, tensor shape: {tensor2.shape}")
print(f"tensor dtype: {tensor2.dtype}, tensor device: {tensor2.device}")
# tensor([[10., 20.],
#        [30., 40.]])
# tensor type: <class 'torch.Tensor'>, tensor shape: torch.Size([2, 2])
# tensor dtype: torch.float32, tensor device: cpu





numpy_data = np.array(list_data)

tensor2_1 = torch.from_numpy(numpy_data)

print(tensor2_1)
print(f"tensor type: {type(tensor2_1)}, tensor shape: {tensor2_1.shape}")
print(f"tensor dtype: {tensor2_1.dtype}, tensor device: {tensor2_1.device}")

tensor2_2 = torch.from_numpy(numpy_data).float()
print('====================================')

print(tensor2_2)
print(f"tensor type: {type(tensor2_2)}, tensor shape: {tensor2_2.shape}")
print(f"tensor dtype: {tensor2_2.dtype}, tensor device: {tensor2_2.device}")
# tensor([[10, 20],
#         [30, 40]])
# tensor type: <class 'torch.Tensor'>, tensor shape: torch.Size([2, 2])
# tensor dtype: torch.int64, tensor device: cpu
# ====================================
# tensor([[10., 20.],
#         [30., 40.]])
# tensor type: <class 'torch.Tensor'>, tensor shape: torch.Size([2, 2])
# tensor dtype: torch.float32, tensor device: cpu





tensor3 = torch.rand(2, 2)
print(tensor3)

tensor4 = torch.randn(2, 2)
print(tensor4)
# tensor([[0.2709, 0.4295],
#         [0.9250, 0.4035]])
# tensor([[ 0.2878, -0.3491],
#         [ 0.9391, -1.0360]])





tensor5 = torch.randn(2, 2)
print(tensor5)

numpy_from_tensor = tensor5.numpy()
print(numpy_from_tensor)
# tensor([[-0.6663, -1.2499],
#         [-1.6641, -0.7628]])
# [[-0.66634035 -1.2498617 ]
#  [-1.664109   -0.7627776 ]]





# Tensor 연산
tensor6 = torch.Tensor([[1, 2, 3], [4, 5, 6]])

tensor7 = torch.Tensor([[7, 8, 9], [10, 11, 12]])

print(tensor6[0])
print(tensor6[:, 1:])
print(tensor7[0:2, 0:-1])
print(tensor7[-1, -1])
print(tensor7[... , -2])
# tensor([1., 2., 3.])
# tensor([[2., 3.],
#         [5., 6.]])
# tensor([[ 7.,  8.],
#         [10., 11.]])
# tensor(12.)
# tensor([ 8., 11.])





tensor8 = tensor6.mul(tensor7)  # tensor8 = tensor6 * tensor7

print(tensor8)
# tensor([[ 7., 16., 27.],
#         [40., 55., 72.]])





# 실행오류
tensor9 = tensor6.matmul(tensor7)   # tensor6 @ tensor7





tensor7.view(3, 2)
# tensor([[ 7.,  8.],
#        [ 9., 10.],
#         [11., 12.]])





tensor9 = tensor6.matmul(tensor7.view(3, 2))  # tensor6 @ tensor7.view(3, 2)

print(tensor9)
# tensor([[ 58.,  64.],
#         [139., 154.]])





# Tensor 합치기(Concatenate)
tensor_cat = torch.cat([tensor6, tensor7])

print(tensor_cat)
# tensor([[ 1.,  2.,  3.],
#         [ 4.,  5.,  6.],
#         [ 7.,  8.,  9.],
#         [10., 11., 12.]])





tensor_cat_dim0 = torch.cat([tensor6, tensor7], dim=0)

print(tensor_cat_dim0)
# tensor([[ 1.,  2.,  3.],
#         [ 4.,  5.,  6.],
#         [ 7.,  8.,  9.],
#         [10., 11., 12.]])





tensor_cat_dim1 = torch.cat([tensor6, tensor7], dim=1)

print(tensor_cat_dim1)
# tensor([[ 1.,  2.,  3.,  7.,  8.,  9.],
#         [ 4.,  5.,  6., 10., 11., 12.]])





# Simple Regression Example
# 데이터 정의
import torch
from torch import nn

x_train = torch.Tensor([1, 2, 3, 4, 5, 6]).view(6,1)
y_train = torch.Tensor([3, 4, 5, 6, 7, 8]).view(6,1)





import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

x_train = torch.Tensor([1, 2, 3, 4, 5, 6]).view(6,1)
y_train = torch.Tensor([3, 4, 5, 6, 7, 8]).view(6,1)

dataset = TensorDataset(x_train, y_train)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# 신경망 모델 구축
class MyNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1,1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    

model = MyNeuralNetwork()

loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)


nums_epoch = 2000

for epoch in range(nums_epoch+1):

    prediction = model(x_train)
    loss = loss_function(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch = ', epoch, ' current loss = ', loss.item())
# epoch =  0  current loss =  23.014493942260742
# epoch =  100  current loss =  0.1645844578742981
# epoch =  200  current loss =  0.07921785861253738
# epoch =  300  current loss =  0.03812915459275246
# epoch =  400  current loss =  0.018352335318922997
# epoch =  500  current loss =  0.00883334968239069
# epoch =  600  current loss =  0.00425164308398962
# epoch =  700  current loss =  0.0020463967230170965
# epoch =  800  current loss =  0.0009849801426753402
# epoch =  900  current loss =  0.00047409351100213826
# epoch =  1000  current loss =  0.00022818443540018052
# epoch =  1100  current loss =  0.0001098292923416011
# epoch =  1200  current loss =  5.2866267651552334e-05
# epoch =  1300  current loss =  2.5444671337027103e-05
# epoch =  1400  current loss =  1.2244702702446375e-05
# epoch =  1500  current loss =  5.894191417610273e-06
# epoch =  1600  current loss =  2.8363731416902738e-06
# epoch =  1700  current loss =  1.3655954944624682e-06
# epoch =  1800  current loss =  6.575176598744292e-07
# epoch =  1900  current loss =  3.1648016829421977e-07
# epoch =  2000  current loss =  1.524490897963915e-07





# 테스트 데이터 예측
x_test = torch.Tensor([-3.1, 3.0, 1.2, -2.5]).view(4,1)

pred = model(x_test)

print(pred)
# tensor([[-1.1015],
#         [ 4.9997],
#         [ 3.1994],
#         [-0.5014]], grad_fn=<AddmmBackward0>)





# 2. MultiVariableLinearRegressionExample
# 데이터 정의
import numpy as np

loaded_data = np.loadtxt('./TrainData.csv', delimiter=',')

x_train_np = loaded_data[ : , 0:-1]

y_train_np = loaded_data[ : , [-1]]

print(loaded_data[:3])
print('========================')
print(x_train_np[:3])
print('========================')
print(y_train_np[:3])
# [[ 1.  2.  0. -4.]
#  [ 5.  4.  3.  4.]
#  [ 1.  2. -1. -6.]]
# ========================
# [[ 1.  2.  0.]
#  [ 5.  4.  3.]
#  [ 1.  2. -1.]]
# ========================
# [[-4.]
#  [ 4.]
#  [-6.]]





import torch
from torch import nn

x_train = torch.Tensor(x_train_np)
y_train = torch.Tensor(y_train_np)





# 신경망 모델 구축
from torch import nn

class MyLinearRegressionModel(nn.Module):

    def __init__(self, input_nodes):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_nodes, 1)
        )

    def forward(self, data):
        prediction = self.linear_stack(data)

        return prediction

model = MyLinearRegressionModel(3)

for name, child in model.named_children():
    for param in child.parameters():
        print(name, param)
# linear_stack Parameter containing:
# tensor([[0.5650, 0.4179, 0.4484]], requires_grad=True)
# linear_stack Parameter containing:
# tensor([0.4568], requires_grad=True)





# 손실함수 및 옵티마이저 설정
loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

loss_list = []
nums_epoch = 2000

for epoch in range(nums_epoch+1):

    prediction = model(x_train)
    loss = loss_function(prediction, y_train)

    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch = ', epoch, ' current loss = ', loss.item())
# epoch =  0  current loss =  21.555883407592773
# epoch =  100  current loss =  0.1591772437095642
# epoch =  200  current loss =  0.0039714002050459385
# epoch =  300  current loss =  0.0009512281394563615
# epoch =  400  current loss =  0.0003183317603543401
# epoch =  500  current loss =  0.00010753283277153969
# epoch =  600  current loss =  3.6332323361421004e-05
# epoch =  700  current loss =  1.2276065717742313e-05
# epoch =  800  current loss =  4.148427706240909e-06
# epoch =  900  current loss =  1.4023293033460504e-06
# epoch =  1000  current loss =  4.741353052395425e-07
# epoch =  1100  current loss =  1.6046881512465916e-07
# epoch =  1200  current loss =  5.436049832496792e-08
# epoch =  1300  current loss =  1.8543675039950358e-08
# epoch =  1400  current loss =  6.268872976278317e-09
# epoch =  1500  current loss =  2.1290835761078597e-09
# epoch =  1600  current loss =  7.501561039013893e-10
# epoch =  1700  current loss =  2.79873069164438e-10
# epoch =  1800  current loss =  1.1481689210501855e-10
# epoch =  1900  current loss =  5.532246535877583e-11
# epoch =  2000  current loss =  3.490454453247693e-11





for name, child in model.named_children():
    for param in child.parameters():
        print(name, param)
# linear_stack Parameter containing:
# tensor([[ 2.0000, -3.0000,  2.0000]], requires_grad=True)
# linear_stack Parameter containing:
# tensor([9.4311e-06], requires_grad=True)





for param in model.parameters():
    print(param)
# Parameter containing:
# tensor([[ 2.0000, -3.0000,  2.0000]], requires_grad=True)
# Parameter containing:
# tensor([9.4311e-06], requires_grad=True)





# 테스트 데이터 예측
x_test = torch.Tensor([ [5, 5, 0], [2, 3, 1], [-1, 0, -1], [10, 5, 2], [4, -1, -2] ])

label = [ 2*data[0] -3*data[1] + 2*data[2]  for data in x_test ]

pred = model(x_test)

print(pred)
print('=============================================')
print(label)
# tensor([[-5.0000],
#         [-3.0000],
#         [-4.0000],
#         [ 9.0000],
#         [ 7.0000]], grad_fn=<AddmmBackward0>)
# =============================================
# [tensor(-5.), tensor(-3.), tensor(-4.), tensor(9.), tensor(7.)]





import matplotlib.pyplot as plt

plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(loss_list, label='train loss')
plt.legend(loc='best')

plt.show()





# 3. LogisticRegressionExample
# 데이터 정의
import numpy as np

loaded_data = np.loadtxt('./diabetes.csv', delimiter=',')

x_train_np = loaded_data[ : , 0:-1]
y_train_np = loaded_data[ : , [-1]]

print('loaded_data.shape = ', loaded_data.shape)
print('x_train_np.shape = ', x_train_np.shape)
print('y_train_np.shape = ', y_train_np.shape)
# loaded_data.shape =  (759, 9)
# x_train_np.shape =  (759, 8)
# y_train_np.shape =  (759, 1)



import torch
from torch import nn

x_train = torch.Tensor(x_train_np)
y_train = torch.Tensor(y_train_np)





# 신경망 모델 구축
class MyLogisticRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.logistic_stack = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        prediction = self.logistic_stack(data)

        return prediction

model = MyLogisticRegressionModel()

for param in model.parameters():
    print(param)
# Parameter containing:
# tensor([[-0.2266, -0.2508, -0.3013,  0.1634,  0.3139,  0.1663,  0.2265, -0.0457]],
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0719], requires_grad=True)


# 손실함수 및 옵티마이저 설정
loss_function = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

train_loss_list = []
train_accuracy_list = []

nums_epoch = 5000

for epoch in range(nums_epoch+1):

    outputs = model(x_train)

    loss = loss_function(outputs, y_train)

    train_loss_list.append(loss.item())

    prediction = outputs > 0.5
    correct = (prediction.float() == y_train)
    accuracy = correct.sum().item() / len(correct)

    train_accuracy_list.append(accuracy)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch = ', epoch, ' current loss = ', loss.item(), ' accuracy = ', accuracy)
# epoch =  0  current loss =  0.682191789150238  accuracy =  0.5797101449275363
# epoch =  100  current loss =  0.5642520189285278  accuracy =  0.6851119894598156
# epoch =  200  current loss =  0.5356192588806152  accuracy =  0.7404479578392622
# epoch =  300  current loss =  0.5180374383926392  accuracy =  0.7562582345191041
# epoch =  400  current loss =  0.5065364241600037  accuracy =  0.7681159420289855
# epoch =  500  current loss =  0.49863582849502563  accuracy =  0.7654808959156785
# epoch =  600  current loss =  0.49300041794776917  accuracy =  0.7681159420289855
# epoch =  700  current loss =  0.4888608157634735  accuracy =  0.7681159420289855
# epoch =  800  current loss =  0.48574694991111755  accuracy =  0.7628458498023716
# epoch =  900  current loss =  0.48335811495780945  accuracy =  0.761528326745718
# epoch =  1000  current loss =  0.48149436712265015  accuracy =  0.766798418972332
# epoch =  1100  current loss =  0.4800184667110443  accuracy =  0.766798418972332
# epoch =  1200  current loss =  0.4788341522216797  accuracy =  0.7707509881422925
# epoch =  1300  current loss =  0.47787222266197205  accuracy =  0.7707509881422925
# epoch =  1400  current loss =  0.47708213329315186  accuracy =  0.7707509881422925
# epoch =  1500  current loss =  0.4764264225959778  accuracy =  0.7707509881422925
# epoch =  1600  current loss =  0.47587695717811584  accuracy =  0.7707509881422925
# epoch =  1700  current loss =  0.47541236877441406  accuracy =  0.769433465085639
# epoch =  1800  current loss =  0.4750160276889801  accuracy =  0.769433465085639
# epoch =  1900  current loss =  0.4746754765510559  accuracy =  0.769433465085639
# epoch =  2000  current loss =  0.4743805229663849  accuracy =  0.769433465085639
# epoch =  2100  current loss =  0.4741232693195343  accuracy =  0.7707509881422925
# epoch =  2200  current loss =  0.4738975465297699  accuracy =  0.7720685111989459
# epoch =  2300  current loss =  0.47369837760925293  accuracy =  0.7720685111989459
# epoch =  2400  current loss =  0.4735215902328491  accuracy =  0.7733860342555995
# epoch =  2500  current loss =  0.473363995552063  accuracy =  0.7733860342555995
# epoch =  2600  current loss =  0.47322285175323486  accuracy =  0.7733860342555995
# epoch =  2700  current loss =  0.4730958342552185  accuracy =  0.7733860342555995
# epoch =  2800  current loss =  0.4729812741279602  accuracy =  0.7733860342555995
# epoch =  2900  current loss =  0.47287747263908386  accuracy =  0.7733860342555995
# epoch =  3000  current loss =  0.47278323769569397  accuracy =  0.7733860342555995
# epoch =  3100  current loss =  0.4726974368095398  accuracy =  0.7733860342555995
# epoch =  3200  current loss =  0.47261908650398254  accuracy =  0.7733860342555995
# epoch =  3300  current loss =  0.4725475013256073  accuracy =  0.7733860342555995
# epoch =  3400  current loss =  0.4724818766117096  accuracy =  0.7733860342555995
# epoch =  3500  current loss =  0.47242164611816406  accuracy =  0.7733860342555995
# epoch =  3600  current loss =  0.4723663032054901  accuracy =  0.7733860342555995
# epoch =  3700  current loss =  0.47231537103652954  accuracy =  0.7707509881422925
# epoch =  3800  current loss =  0.47226840257644653  accuracy =  0.7707509881422925
# epoch =  3900  current loss =  0.4722250699996948  accuracy =  0.7707509881422925
# epoch =  4000  current loss =  0.4721851050853729  accuracy =  0.7707509881422925
# epoch =  4100  current loss =  0.4721481502056122  accuracy =  0.7707509881422925
# epoch =  4200  current loss =  0.4721139967441559  accuracy =  0.7707509881422925
# epoch =  4300  current loss =  0.4720824062824249  accuracy =  0.7707509881422925
# epoch =  4400  current loss =  0.47205308079719543  accuracy =  0.7707509881422925
# epoch =  4500  current loss =  0.472025990486145  accuracy =  0.7707509881422925
# epoch =  4600  current loss =  0.4720008671283722  accuracy =  0.7707509881422925
# epoch =  4700  current loss =  0.4719775915145874  accuracy =  0.7707509881422925
# epoch =  4800  current loss =  0.47195595502853394  accuracy =  0.769433465085639
# epoch =  4900  current loss =  0.4719359576702118  accuracy =  0.769433465085639
# epoch =  5000  current loss =  0.4719173014163971  accuracy =  0.769433465085639





for name, child in model.named_children():
    for param in child.parameters():
        print(name, param)
# logistic_stack Parameter containing:
# tensor([[-0.8934, -3.5666,  0.2699, -0.5937, -0.3212, -2.4303, -0.9964, -0.0779]],
#        requires_grad=True)
# logistic_stack Parameter containing:
# tensor([0.1699], requires_grad=True)





for param in model.parameters():
    print(param)
# Parameter containing:
# tensor([[-0.8934, -3.5666,  0.2699, -0.5937, -0.3212, -2.4303, -0.9964, -0.0779]],
#        requires_grad=True)
# Parameter containing:
# tensor([0.1699], requires_grad=True)





# 손실 및 정확도 추세
import matplotlib.pyplot as plt

plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(train_loss_list, label='train loss')
plt.legend(loc='best')

plt.show()





import matplotlib.pyplot as plt

plt.title('Accuracy Trend')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(train_accuracy_list, label='train accuracy')
plt.legend(loc='best')

plt.show()