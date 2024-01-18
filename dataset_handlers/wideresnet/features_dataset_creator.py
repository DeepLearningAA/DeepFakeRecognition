import os
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split, DataLoader

wide_resnet50_2 = models.wide_resnet50_2(weights='Wide_ResNet50_2_Weights.IMAGENET1K_V1')
wide_resnet50_2.requires_grad_(False)

model_general_features = nn.Sequential(
    wide_resnet50_2.conv1,
    wide_resnet50_2.bn1,
    wide_resnet50_2.relu,
    wide_resnet50_2.maxpool,
    wide_resnet50_2.layer1,
    wide_resnet50_2.layer2,
    wide_resnet50_2.layer3,
    nn.Flatten()
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.299, 0.224, 0.225],
        
    )
])

data_augmentation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAutocontrast(),
    transforms.RandomRotation(20),
    transforms.RandomEqualize(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.299, 0.224, 0.225],
    )
])

k = 1

for fold in range(k):
    for j in range(3):
        train_dataset = datasets.ImageFolder(
            root='../../data/splitted/train',
            transform=data_augmentation
        )

        cur = len(train_dataset)

        train_fake_path = f'../../data/splitted/train_widerestnet_data_augmentation/train_{fold}/fake'
        train_real_path = f'../../data/splitted/train_widerestnet_data_augmentation/train_{fold}/real'

        dirs = [train_fake_path, train_real_path]
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

        for i, (image, label) in enumerate(train_dataset):
            if label == 0:
                print(f'Processing image {(j+1)*cur + i} with label {label} into path {train_fake_path}')
                with open(f'{train_fake_path}/{(j+1)*cur + i}.flatten', 'wb') as f:
                    torch.save(model_general_features(image.unsqueeze(0)), f)

            if label == 1:
                print(f'Processing image {(j+1)*cur + i} with label {label} into path {train_real_path}')
                with open(f'{train_real_path}/{(j+1)*cur + i}.flatten', 'wb') as f:
                    torch.save(model_general_features(image.unsqueeze(0)), f)


test_dataset = datasets.ImageFolder(
    root='../../data/splitted/test',
    transform=transform
)

test_fake_path = f'../../data/splitted/test_widerestnet/test/fake'
test_real_path = f'../../data/splitted/test_widerestnet/test/real'

dirs = [test_fake_path, test_real_path]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

for i, (image, label) in enumerate(test_dataset):
    if label == 0:
        print(f'Processing image {i} with label {label} into path {test_fake_path}')
        with open(f'{test_fake_path}/{i}.flatten', 'wb') as f:
            torch.save(model_general_features(image.unsqueeze(0)), f)

    if label == 1:
        print(f'Processing image {i} with label {label} into path {test_real_path}')
        with open(f'{test_real_path}/{i}.flatten', 'wb') as f:
            torch.save(model_general_features(image.unsqueeze(0)), f)