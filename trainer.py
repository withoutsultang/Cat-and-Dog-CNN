# trainer.py

import torch
import numpy as np
from copy import deepcopy

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
