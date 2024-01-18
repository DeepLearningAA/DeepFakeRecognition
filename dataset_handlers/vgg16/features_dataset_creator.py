import os
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split, DataLoader
from PIL import Image

vgg16 = models.vgg16(weights=True)
vgg16.requires_grad_(False)


starting_layer = 24
k = 5

# Getting first 24 feature layers of VGG16
vgg16_general_features = nn.Sequential(
    vgg16.features[:starting_layer],
    nn.Flatten()
)

# Dataset definition
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
    transforms.RandomEqualize(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.299, 0.224, 0.225],
        
    )
])

for fold in range(k):
    dataset = datasets.ImageFolder(
        root='../../data/real_and_fake_face',
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataset.dataset.transform = data_augmentation


    # Saving to feature dataset
    train_fake_path = f'../../data/real_and_fake_vgg16_data_augmentation/train_{fold}/fake'
    train_real_path = f'../../data/real_and_fake_vgg16_data_augmentation/train_{fold}/real'

    test_fake_path = f'../../data/real_and_fake_vgg16_data_augmentation/test_{fold}/fake'
    test_real_path = f'../../data/real_and_fake_vgg16_data_augmentation/test_{fold}/real'

    dirs = [train_fake_path, train_real_path, test_fake_path, test_real_path]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    for i, (image, label) in enumerate(train_dataset):
        if label == 0:
            print(f'Processing image {i} with label {label} into path {train_fake_path}')
            with open(f'{train_fake_path}/{i}.flatten', 'wb') as f:
                torch.save(vgg16_general_features(image.unsqueeze(0)), f)

        if label == 1:
            print(f'Processing image {i} with label {label} into path {train_real_path}')
            with open(f'{train_real_path}/{i}.flatten', 'wb') as f:
                torch.save(vgg16_general_features(image.unsqueeze(0)), f)


    for i, (image, label) in enumerate(test_dataset):
        if label == 0:
            print(f'Processing image {i} with label {label} into path {test_fake_path}')
            with open(f'{test_fake_path}/{i}.flatten', 'wb') as f:
                torch.save(vgg16_general_features(image.unsqueeze(0)), f)

        if label == 1:
            print(f'Processing image {i} with label {label} into path {test_real_path}')
            with open(f'{test_real_path}/{i}.flatten', 'wb') as f:
                torch.save(vgg16_general_features(image.unsqueeze(0)), f)
