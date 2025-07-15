# 3. albumentations
from glob import glob
from PIL import Image
import random

data_root = './cats_vs_dogs_small/'
filepath_list = list(glob(f"{data_root}/**/**/*.jpg"))
random.shuffle(filepath_list)

image_list = []
for idx in range(3):
    image_list.append(Image.open(filepath_list[idx]))



image_list[0]



# pip install albumentations



# 이미지 변환 테스트
import matplotlib.pyplot as plt
'''
    이미지 시각화 함수를 정의한다.
'''

def visualize_images(image1, image2):
    '''
        :param images: cv2(ndarray) 이미지 리스트
        :param classes: 클래스 리스트
        :return: None
    '''
    # 4x2의 그리드 생성 (바둑판 이미지 틀 생성)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image1)
    axs[0].set_title('original')

    axs[1].imshow(image2)
    axs[1].set_title('transformed')

    plt.tight_layout()
    plt.show()



import albumentations as A
import numpy as np

sample_image = image_list[0]



'''
    좌우 반전 테스트
'''
transform = A.HorizontalFlip(p=1.0)
transformed_image = transform(image=np.array(sample_image))['image']
visualize_images(sample_image, transformed_image)



'''
    형태 변환 테스트
'''
# transform = A.ShiftScaleRotate(p=0.5, border_mode=0)
# transform = A.OpticalDistortion(p=1.0, distort_limit=1.0)
transform = A.GridDistortion(p=1.0)
transformed_image = transform(image=np.array(sample_image))['image']
visualize_images(sample_image, transformed_image)



'''
    Blur 테스트
'''
transform = A.MotionBlur(p=1.0, blur_limit=(3, 5))
# transform = A.Blur(p=1.0, blur_limit=(3, 5))
transformed_image = transform(image=np.array(sample_image))['image']
visualize_images(sample_image, transformed_image)



'''
    픽셀 값 변환
'''
# transform = A.CLAHE(p=1.0)
# transform = A.ChannelShuffle(p=1.0)
transform = A.ColorJitter(p=1.0)
transformed_image = transform(image=np.array(sample_image))['image']
visualize_images(sample_image, transformed_image)



'''
    Dropout 테스트
'''
transform = A.ChannelDropout(p=1.0)
transformed_image = transform(image=np.array(sample_image))['image']
visualize_images(sample_image, transformed_image)



'''
    여러 transform 모듈을 하나로 합친다.
'''
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.CLAHE(p=0.5),
    A.Blur(p=0.5),
    A.ShiftScaleRotate(p=0.5, border_mode=0),
    A.OpticalDistortion(),
    A.GridDistortion(),
    A.HueSaturationValue(),
    A.Resize(height=224, width=224),
    # A.Normalize(p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ## 이미지 픽셀 값 정규화
    # ToTensorV2() ## 모델에 입력할 때 사용
])
random.seed(42)

## Dataset 클래스에서 transform을 적용할 때 아래와 같은 코드로 변경 해야함
transformed_image = transform(image=np.array(sample_image))['image']

'''
    A.Normalize와 ToTensorV2를 적용했을 경우 아래의 시각화가 되지 않음.
'''
visualize_images(sample_image, transformed_image)



# 이미지 역변환(역정규화)
'''
    변환된 이미지 픽셀 값 확인
'''
transformed_image



import torch
from torchvision.transforms.functional import to_pil_image

def denormalize(tensor, mean, std):
    '''
        텐서를 역정규화 한다.
    '''
    mean = torch.tensor(mean).reshape(-1, 1, 1)
    std = torch.tensor(std).reshape(-1, 1, 1)

    # 역정규화 수행
    tensor = tensor * std + mean

    # 텐서의 값 범위를 0과 1 사이로 조정합니다.
    tensor = torch.clamp(tensor, 0, 1)

    return tensor


def tensor_to_pil(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    '''
        정규화된 텐서를 Pillow 이미지로 변환한다.
    '''
    # 역정규화
    tensor = denormalize(tensor, mean, std)

    # 텐서를 Pillow 이미지로 변환한다.
    pil_image = to_pil_image(tensor)

    return pil_image



'''
    역정규화 한 이미지를 확인한다.
'''

from albumentations.pytorch import ToTensorV2
import albumentations as A

normalize_mean = (0.485, 0.456, 0.406)
normalize_std  = (0.229, 0.224, 0.225)

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=normalize_mean, std=normalize_std),
    ToTensorV2()
])

out = transform(image=np.array(sample_image))['image']   # torch.Tensor, 0~1 정규화
recover = denormalize(out, normalize_mean, normalize_std)  # 여기서만 denormalize
img_pil = to_pil_image(recover)
visualize_images(np.array(sample_image), np.array(img_pil))