# 머신러닝 기말고사 프로젝트 - 고양이와 강아지 이미지 분류
이 프로젝트는 고양이와 강아지 이미지를 분류하는 문제를 해결하기 위해 머신러닝 모델을 구현했습니다. Kaggle 데이터셋을 사용하여 여러 모델(VGG-16, ResNet-18)을 비교하고 최적의 성능을 도출하였습니다.

## 데이터셋 및 프로젝트 구조
- 데이터셋 출처: [Kaggle Cat and Dog Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
- 데이터셋은 Google Drive를 사용해 로드했습니다.
- 프로젝트는 주피터 노트북을 통해 진행하였으며, 주요 단계는 데이터 로드, 데이터 전처리, 모델 학습, 성능 평가입니다.
- 고양이와 강아지 이미지 총 8,067장

## 주요 라이브러리
- **NumPy**: 벡터 및 행렬 연산을 위한 라이브러리
- **Matplotlib**: 데이터 시각화를 위한 라이브러리
- **Torch 및 TorchVision**: 딥러닝 모델 구현 및 전처리
- **PIL (Pillow)**: 이미지 처리 및 로딩
- **scikit-learn**: 모델 평가를 위한 혼동 행렬 등

```python
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from copy import deepcopy
```
---

## 1. 구글 드라이브 연결 및 데이터 로드

```python
# Google Drive를 마운트하여 데이터셋을 Colab 환경에 로드
from google.colab import drive
drive.mount('/content/drive')

# 데이터 경로 설정
root = '/content/drive/MyDrive/Colab Notebooks/content/archive'
train_path = root + '/training_set/training_set'
test_path = root + '/test_set/test_set'

# 데이터 로드 및 확인을 위한 샘플 이미지 시각화
img = Image.open(train_path + "/cats/cat.34.jpg")
plt.imshow(np.array(img))
plt.show()
```
![image](https://github.com/user-attachments/assets/320521d6-0ddd-4267-b13d-484e806fb32f)

### 설명
- Google Drive에 데이터를 저장한 후 연결하여 데이터셋 로드
- 이미지(고양이)를 시각화하여 데이터가 올바르게 로드되었는지 확인

## 2. 데이터셋 클래스 정의

```python
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

# 데이터셋 인스턴스 생성
train = CatandDog(train_path, transform=transform)
test = CatandDog(test_path, transform=transform)
```

### 설명
- `CatandDog` 클래스는 PyTorch `Dataset` 클래스를 상속받아 데이터셋을 정의
- 학습에 사용할 데이터를 로드하고 레이블을 할당 (고양이: 0, 강아지: 1).
- 데이터 전처리를 위해 `Resize`, `ToTensor`, `Normalize` 적용

---

## 3. VGG-16 및 ResNet-18 모델 정의
딥러닝 아키텍처로 VGG-16과 ResNet-18(미리 정의된 모델)을 사용하여 모델을 구현

### VGG-16 모델
```python
class VGG16(nn.Module):
  def __init__(self, num_classes=1):
    super(VGG16, self).__init__()

    self.features = nn.Sequential(
        # Conv Block 1
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Conv Block 2
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Conv Block 3
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Conv Block 4
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Conv Block 5
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    # 분류를 위한 Fully Connected Layer
    self.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),  # 드롭아웃을 적용하여 과적합 방지
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, num_classes),
        nn.Sigmoid()  # 이진 분류를 위한 시그모이드 함수
    )

  def forward(self, x):
      #  최종 출력 반환
      x = self.features(x)
      x = x.view(x.size(0), -1)  # 평탄화
      x = self.classifier(x)
      return x
```

### ResNet-18 모델
```python
class ResNet18Binary(nn.Module):
    def __init__(self):
        super(ResNet18Binary, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # 사전 학습된 ResNet-18 사용
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # 드롭아웃 추가 (과적합 방지)
            nn.Linear(512, 1),
            nn.Sigmoid() # 이진 분류를 위한 시그모이드 함수
        )

    def forward(self, x):
        return self.resnet(x)
```

### 설명
- VGG-16과 ResNet-18 모델을 커스터마이징(BCELoss)하여 **이진 분류** 문제에 맞게 설정
- 최종 레이어는 **시그모이드(Sigmoid)** 활성화 함수를 사용

---

## 4. 모델 학습을 위한 Trainer 클래스

```python
class Trainer:
    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

    def train(self, train_loader, valid_loader, config):
        best_loss = float('inf')

        for epoch in range(config.n_epochs):
            train_loss, train_acc = self._train(train_loader, config)
            valid_loss, valid_acc = self._validate(valid_loader, config)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model_state = deepcopy(self.model.state_dict()) # 모델 저장
            # 학습 결과 출력
            print(f"Epoch({epoch+1}/{config.n_epochs}): train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, best_loss={best_loss:.4f}, train_acc={train_acc*100:.2f}, valid_acc={valid_acc*100:.2f}")

        self.model.load_state_dict(best_model_state)

    def _train(self, loader, config):
        self.model.train() # 모델을 학습 모드로 설정
        total_loss, correct, total = 0, 0, 0

        for x, y in loader:
            x, y = x.to(config.device), y.to(config.device).float().view(-1, 1)
            y_hat = self.model(x) # 모델의 예측값 계산
            loss = self.crit(y_hat, y) # 손실 계산

            self.optimizer.zero_grad() # 기울기 초기화
            loss.backward() # 역전파 수행
            self.optimizer.step() # 가중치 업데이트

            total_loss += loss.item()
            correct += (torch.round(y_hat) == y).sum().item() # 정확도 계산
            total += y.size(0)

        return total_loss / len(loader), correct / total

  def _validate(self, loader, config):
    self.model.eval() # 모델을 평가 모드로 설정
    total_loss = 0
    corr = 0
    total = 0

    with torch.no_grad():
      for x,y in loader:
        x, y = x.to(config.device), y.to(config.device).float().view(-1,1)
        y_hat = self.model(x)
        loss = self.crit(y_hat, y)
        total_loss += loss.item()

        predicted = (y_hat > 0.5).float()
        corr = (predicted == y).sum().item()
        total = y.size(0)

      avg_loss = total_loss / len(loader)
      accuracy = corr / total

    return avg_loss, accuracy
```

### 설명
- `Trainer` 클래스는 학습과 검증을 수행하는 기능을 포함하고, `train` 및 `validate` 메서드로 모델을 학습

---
## 5. 모델 인자 설정 및 Main
```python
def define_arg():
    p = argparse.ArgumentParser()

    p.add_argument("--model_fn", required=False, default="/content/model_resnet18_bce.pth")
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--verbose", type=int, default=1)

    config = p.parse_args(args=[])
    config.device = torch.device("cuda:%d" % config.gpu_id if config.gpu_id >= 0 else "cpu")

    return config

def main(config):
    # 학습 및 검증 데이터셋 분할 (80:20)
    train_size = int(len(reduced) * 0.8)
    valid_size = len(reduced) - train_size
    train_data, valid_data = torch.utils.data.random_split(reduced, [train_size, valid_size])

    # 데이터 로더 생성
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=True)

    # 모델, 손실 함수, 옵티마이저 정의
    model = ResNet18Binary().to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    crit = nn.BCELoss()

    # Trainer 클래스 생성 및 학습 실행
    trainer = Trainer(model, optimizer, crit)
    trainer.train(train_loader, valid_loader, config)

    # 학습된 모델 저장
    torch.save({
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "config": config,
    }, config.model_fn)

# Main 함수 실행
if __name__ == "__main__":
    config = define_arg()
    main(config)
```
### 설명
- `define_arg()`는 에포크 수, 배치 크기, 모델 저장 경로 설정
- `main()`은 데이터셋 분할, 모델 초기화, 학습 실행 및 모델 저장을 담당
---
## 6. 성능 평가 및 시각화
훈련된 모델의 성능을 평가하기 위해 **훈련/검증 손실 곡선**, **혼동 행렬** 출력

### 손실 및 정확도 그래프
```python
def plot_training_history(train_losses, valid_losses, train_accuracies, valid_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 5))
    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training loss')
    plt.plot(epochs, valid_losses, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, valid_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
```
![image](https://github.com/user-attachments/assets/d199d39e-0ee3-479d-80e0-25cfdf122446)

### 혼동 행렬
```python
def plot_confusion_matrix(model, data_loader, device):
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.round(outputs).cpu().numpy()
            labels = labels.cpu().numpy()

            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cat', 'Dog'])
    disp.plot(cmap='Blues')
    plt.show()
```
![image](https://github.com/user-attachments/assets/d004332c-c3e5-401f-b61e-52ed9590a250)

### 설명
- 손실 및 정확도 곡선을 통해 모델이 학습되는 과정을 시각화하고, 혼동 행렬을 통해 클래스 간 오분류를 확인

---

## 결론
- 이번 프로젝트에서는 고양이와 강아지 이미지를 분류하는 이진 분류 모델을 구현하였습니다.
- **VGG-16과 ResNet-18**을 비교한 결과, **ResNet-18** 모델이 더 좋은 성능인걸 알 수 있었습니다.
- 데이터에 과적합이 있을때 dropout 파라미터가 모델에 미치는 영향을 이해할 수 있었습니다.
- 머신러닝 기반 이미지 분류 모델이 실제 이미지 데이터를 효과적으로 분류하는지 확인할 수 있었습니다.  

## 향후 개선 방향
훈련 데이터의 일부 과적합을 완전히 해결하지 못했는데 다음과 같은 개선 방향을 생각했습니다.

- 데이터 증강 또는 dropout 수정 등으로  **일반화 성능** 향상
- 하이퍼파라미터 최적화를 위해 **Grid Search** 도입
- 다른 전이 학습 모델 사용
