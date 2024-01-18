from torch.utils.data import Dataset
import torch
import os

class FeatureDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.classes = sorted(os.listdir(root_path))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = self._load_samples()
        self.transform = transform


    def _load_samples(self):
        samples = []
        for class_name, class_idx in self.class_to_idx.items():
            class_path = os.path.join(self.root_path, class_name)
            for file in os.listdir(class_path):
                if file.endswith('.flatten'):
                    tensor_path = os.path.join(class_path, file)
                    samples.append((tensor_path, class_idx))
        return samples
    

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        tensor_path, class_idx = self.samples[idx]
        tensor = torch.load(tensor_path).reshape(512)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, torch.tensor(class_idx)
    

class EasyDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.classes = sorted(os.listdir(root_path))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = self._load_samples()
        self.transform = transform


    def _load_samples(self):
        samples = []
        for class_name, class_idx in self.class_to_idx.items():
            class_path = os.path.join(self.root_path, class_name)
            for file in os.listdir(class_path):

                easy = file.split('_')[0] == 'easy' or file.split('_')[0] == 'real'

                if file.endswith('.flatten') and easy:
                    tensor_path = os.path.join(class_path, file)
                    samples.append((tensor_path, class_idx))
        return samples
    

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        tensor_path, class_idx = self.samples[idx]
        tensor = torch.load(tensor_path).reshape(1,512)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, torch.tensor(class_idx)