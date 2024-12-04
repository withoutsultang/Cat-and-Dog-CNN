# utils.py

import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 데이터셋 클래스 정의
class CatandDog(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        # 고양이와 강아지 폴더의 이미지를 리스트에 추가
        for label in ['cats', 'dogs']:
            path = os.path.join(root, label)
            for image in os.listdir(path):
                if image != '_DS_Store': # 시스템 파일 무
                    self.images.append(os.path.join(path, image))

    def __len__(self):
        # 데이터셋의 총 이미지 개수 반환
        return len(self.images)

    def __getitem__(self, idx):
        # 주어진 인덱스의 이미지와 레이블 반환
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = 1.0 if 'dog' in self.images[idx].lower() else 0.0
        # 파일명에 따라 레이블 결정 (강아지: 1, 고양이: 0)

        if self.transform:
            image = self.transform(image)

        return image, label

# 데이터 전처리 (이미지 크기 조정, 텐서 변환, 정규화)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
