# train.py

import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from torch import nn
from utils import CatandDog, transform
from trainer import Trainer
from model import ResNet18Binary

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
