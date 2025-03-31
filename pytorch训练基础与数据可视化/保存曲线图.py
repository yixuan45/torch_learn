# -*- coding: utf-8 -*-
import wandb
import random
import datetime

runtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
wandb.init(
    project='wandb_demo',
    name=f"run={runtime}",
    config={
        "learning_rate": 0.05, "model": "CNN", "dataset": "MNIST", "epochs": 10,
    }
)

offset = random.random() / 2
epochs = 5
for epoch_i in range(2, epochs):
    acc = 1 - 2 ** -epoch_i - random.random() / epoch_i - offset
    loss = 2 ** -epoch_i - random.random() / epoch_i - offset
    # 记录指标
    wandb.log({"acc": acc, "loss": loss})

wandb.finish()
