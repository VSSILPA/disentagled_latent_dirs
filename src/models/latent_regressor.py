import torch.nn as nn
import torch.functional as F
from catalyst import dl


class Encoder(nn.Module):
    def __init__(self, latent_dimension, backbone="resnet18", f_size=256, **bb_kwargs):
        super().__init__()

        features = 512 if "resnet" in backbone else 1024

        self.backbone = nn.Sequential(
            globals()[backbone](**bb_kwargs),
            nn.ReLU(),
            nn.Linear(features, f_size),
            nn.ReLU(),
        )
        self.latent_head = nn.Linear(f_size, latent_dimension)

    def forward(self, x):
        features = self.backbone(x)
        latent = self.latent_head(features)
        return latent


class LatentRunner(dl.Runner):
    def predict_batch(self, batch):
        x, y = batch
        return self.model(x.to(self.device))

    def _handle_batch(self, batch):
        # model train/valid step
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y, y_hat)

        loss_dict = {"loss": loss}

        for i in range(y.shape[1]):
            loss_dict["factor_{}".format(i)] = F.mse_loss(y[:, i], y_hat[:, i])

        self.batch_metrics.update(loss_dict)

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()