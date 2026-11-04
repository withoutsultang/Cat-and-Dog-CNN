# train.py

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from torch import nn
from utils import CatandDog, transform
from trainer import Trainer
from model import ResNet18Binary

def define_arg():
    p = argparse.ArgumentParser()

    p.add_argument("--data-dir", type=Path, required=True, help="Directory containing cats/ and dogs/")
    p.add_argument("--model_fn", type=Path, default=Path("outputs/resnet18.pth"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--verbose", type=int, default=1)

    config = p.parse_args()
    if config.n_epochs < 1 or config.batch_size < 1:
        p.error("--n_epochs and --batch_size must be positive")
    config.device = torch.device("cuda:%d" % config.gpu_id if config.gpu_id >= 0 else "cpu")

    return config

def main(config):
    # 학습 및 검증 데이터셋 분할 (80:20)
    torch.manual_seed(config.seed)
    dataset = CatandDog(config.data_dir, transform)
    if len(dataset) < 2:
        raise ValueError("Training requires at least two images")
    train_size = int(len(dataset) * 0.8)
    train_data, valid_data = random_split(
        dataset, [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    # 데이터 로더 생성
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False)

    # 모델, 손실 함수, 옵티마이저 정의
    model = ResNet18Binary(pretrained=not config.no_pretrained).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    crit = nn.BCELoss()

    # Trainer 클래스 생성 및 학습 실행
    trainer = Trainer(model, optimizer, crit)
    trainer.train(train_loader, valid_loader, config)

    # 학습된 모델 저장
    config.model_fn.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": trainer.model.state_dict(),
        "config": {**vars(config), "device": str(config.device),
                   "data_dir": str(config.data_dir), "model_fn": str(config.model_fn)},
    }, config.model_fn)

# Main 함수 실행
if __name__ == "__main__":
    config = define_arg()
    main(config)
