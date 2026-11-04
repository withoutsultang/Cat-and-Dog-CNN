# utils.py

import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# 데이터셋 클래스 정의
class CatandDog(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        # 고양이와 강아지 폴더의 이미지를 리스트에 추가
        for label, folder in enumerate(("cats", "dogs")):
            path = os.path.join(root, folder)
            for image in sorted(os.listdir(path)):
                image_path = os.path.join(path, image)
                if os.path.isfile(image_path) and image.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.images.append((image_path, float(label)))

    def __len__(self):
        # 데이터셋의 총 이미지 개수 반환
        return len(self.images)

    def __getitem__(self, idx):
        # 주어진 인덱스의 이미지와 레이블 반환
        image_path, label = self.images[idx]
        with Image.open(image_path) as source:
            image = source.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# 데이터 전처리 (이미지 크기 조정, 텐서 변환, 정규화)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
