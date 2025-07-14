# 1. TransferLearning_Basic
import torch
from torchvision import models

pretrained_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

print(pretrained_model)
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )



class MyTransferLearningModel(torch.nn.Module):

    def __init__(self, pretrained_model, feature_extractor):
        super().__init__()

        if (feature_extractor):
            for param in pretrained_model.parameters():
                param.requires_grad = False

        pretrained_model.classifier = torch.nn.Sequential(   # 512* 7*7 = 25088
            torch.nn.Linear(pretrained_model.classifier[0].in_features, 128),   # 25088 -> 128
            torch.nn.Linear(128, 2)   # 128 -> 2 (이진분류) -> softmax 사용 ---> crossentropyloss 정의 예정
        )

        self.model = pretrained_model

    def forward(self, data):
        logits = self.model(data)
        return logits
    


feature_extractor = True  # True: Feature Extractor,  False: Fine Tuning

model = MyTransferLearningModel(pretrained_model, feature_extractor)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-6)

loss_function = torch.nn.CrossEntropyLoss()
print(model)
# MyTransferLearningModel(
#   (model): VGG(
#     (features): Sequential(
#       (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (1): ReLU(inplace=True)
#       (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (3): ReLU(inplace=True)
#       (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#       (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (6): ReLU(inplace=True)
#       (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (8): ReLU(inplace=True)
#       (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#       (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (11): ReLU(inplace=True)
#       (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (13): ReLU(inplace=True)
#       (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (15): ReLU(inplace=True)
#       (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#       (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (18): ReLU(inplace=True)
#       (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (20): ReLU(inplace=True)
#       (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (22): ReLU(inplace=True)
#       (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#       (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (25): ReLU(inplace=True)
#       (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (27): ReLU(inplace=True)
#       (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (29): ReLU(inplace=True)
#       (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     )
#     (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#     (classifier): Sequential(
#       (0): Linear(in_features=25088, out_features=128, bias=True)
#       (1): Linear(in_features=128, out_features=2, bias=True)
#     )
#   )
# )





# 2. FineTuning_CatsDogs [GPU 요구되므로 colab에서 실행]
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, random_split



import os

ROOT_DIR = '/content'

DATA_ROOT_DIR = os.path.join(ROOT_DIR, 'cats_and_dogs_filtered')

TRAIN_DATA_ROOT_DIR = os.path.join(DATA_ROOT_DIR, 'train')

VALIDATION_DATA_ROOT_DIR = os.path.join(DATA_ROOT_DIR, 'validation')
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"using PyTorch version: {torch.__version__}, Device: {DEVICE}")
# using PyTorch version: 2.6.0+cu124, Device: cuda



# 데이터 다운로드
# !wget https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip

# --2025-07-14 13:03:43--  https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
# Resolving storage.googleapis.com (storage.googleapis.com)... 142.251.107.207, 74.125.196.207, 108.177.11.207, ...
# Connecting to storage.googleapis.com (storage.googleapis.com)|142.251.107.207|:443... connected.
# HTTP request sent, awaiting response... 200 OK
# Length: 68606236 (65M) [application/zip]
# Saving to: ‘cats_and_dogs_filtered.zip’
# 
# cats_and_dogs_filte 100%[===================>]  65.43M   244MB/s    in 0.3s    
# 
# 2025-07-14 13:03:44 (244 MB/s) - ‘cats_and_dogs_filtered.zip’ saved [68606236/68606236]



import os
import shutil

if os.path.exists('/content/cats_and_dogs_filtered/'):    # 작업 디렉토리는 cats_and_dogs_filtered

    shutil.rmtree('/content/cats_and_dogs_filtered/')
    print('/content/cats_and_dogs_filtered/  is removed !!!')



# 압축파일 풀기

import zipfile

with zipfile.ZipFile('/content/cats_and_dogs_filtered.zip', 'r') as target_file:

    target_file.extractall('/content/')



import os

# train data 개수

train_cats_list = os.listdir('/content/cats_and_dogs_filtered/train/cats/')

train_dogs_list = os.listdir('/content/cats_and_dogs_filtered/train/dogs/')

# validation data 개수

test_cats_list = os.listdir('/content/cats_and_dogs_filtered/validation/cats/')

test_dogs_list = os.listdir('/content/cats_and_dogs_filtered/validation/dogs/')

print(len(train_cats_list), len(train_dogs_list))

print(len(test_cats_list), len(test_dogs_list))
# 1000 1000
# 500 500



train_config = transforms.Compose([transforms.Resize((224,224)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor()])

validation_config = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor()])



train_dataset = datasets.ImageFolder('/content/cats_and_dogs_filtered/train/', train_config)

validation_dataset = datasets.ImageFolder('/content/cats_and_dogs_filtered/validation/', validation_config)

test_dataset = datasets.ImageFolder('/content/cats_and_dogs_filtered/validation/', validation_config)



BATCH_SIZE = 32

train_dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

validation_dataset_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



# 1개의 배치를 추출
images, labels = next(iter(train_dataset_loader))



import matplotlib.pyplot as plt

# ImageFolder의 속성 값인 class_to_idx를 할당

labels_map = { v:k  for k, v in train_dataset.class_to_idx.items() }

figure = plt.figure(figsize=(6, 7))

cols, rows = 4, 4

# 이미지 출력

for i in range(1, cols*rows+1):

    sample_idx = torch.randint(len(images), size=(1,)).item()
    img, label = images[sample_idx], labels[sample_idx].item()

    figure.add_subplot(rows, cols, i)

    plt.title(labels_map[label])
    plt.axis("off")

    # 본래 이미지의 shape은 (3, 224, 224) 인데,
    # 이를 imshow() 함수로 이미지 시각화 하기 위하여 (224, 224, 3)으로 shape 변경을 한 후 시각화
    plt.imshow(torch.permute(img, (1, 2, 0)))

plt.show()



pretrained_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
print(pretrained_model)
# Downloading: "https://download.pytorch.org/models/vit_b_16-c867db91.pth" to /root/.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth
# 100%|██████████| 330M/330M [00:01<00:00, 180MB/s]
# VisionTransformer(
#   (conv_proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
#   (encoder): Encoder(
#     (dropout): Dropout(p=0.0, inplace=False)
#     (layers): Sequential(
#       (encoder_layer_0): EncoderBlock(
#         (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#         (self_attention): MultiheadAttention(
#           (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
#         )
#         (dropout): Dropout(p=0.0, inplace=False)
#         (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#         (mlp): MLPBlock(
#           (0): Linear(in_features=768, out_features=3072, bias=True)
#           (1): GELU(approximate='none')
#           (2): Dropout(p=0.0, inplace=False)
#           (3): Linear(in_features=3072, out_features=768, bias=True)
#           (4): Dropout(p=0.0, inplace=False)
#         )
#       )
#===========  (...) layer 1-10 생략 ================
#       (encoder_layer_11): EncoderBlock(
#         (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#         (self_attention): MultiheadAttention(
#           (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
#         )
#         (dropout): Dropout(p=0.0, inplace=False)
#         (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#         (mlp): MLPBlock(
#           (0): Linear(in_features=768, out_features=3072, bias=True)
#           (1): GELU(approximate='none')
#           (2): Dropout(p=0.0, inplace=False)
#           (3): Linear(in_features=3072, out_features=768, bias=True)
#           (4): Dropout(p=0.0, inplace=False)
#         )
#       )
#     )
#     (ln): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#   )
#   (heads): Sequential(
#     (head): Linear(in_features=768, out_features=1000, bias=True)
#   )
# )
