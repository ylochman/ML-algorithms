import itertools

import torch
import torch.nn as nn
import torch.utils.data
import tqdm
import torch.functional as F
import numpy as np
from tensorboardX import SummaryWriter

from src.losses import SoftDiceLoss
from src.score import score_function_fast
from src.utils import save_checkpoint

CHECKPOINT_STEP = 400


class Trainer:
    def __init__(self, net, config, limit=None, writer=None):
        net.train()
        net.to(config['DEVICE'])
        self.device = config['DEVICE']
        self.net = net
        self.config = config
        self.loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.15, 1, 1]).to(config['DEVICE']))
        # self.loss = SoftDiceLoss(n_classes=3)
        self.optimizer = torch.optim.Adam(net.parameters(), lr=config['LR'])
        if writer is None:
            self.tensorboard = SummaryWriter()
        else:
            self.tensorboard = writer
        self.limit = limit

    def run(self, dataloader, epochs=1, start_epoch=-1):
        print(">> Running trainer")
        for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
            print(">>> Epoch %s" % epoch)
            for idx, (image, target) in enumerate(tqdm.tqdm(dataloader, ascii=True)):
                if self.limit is not None and idx % self.limit == self.limit - 1:
                    break
                image, target = image.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                predict = self.net(image)
                loss = self.loss(predict, target)
                batch_prediction = torch.argmax(torch.softmax(predict.cpu(), 1), 1, keepdim=True)
                score = score_function_fast(batch_prediction.numpy(), target.cpu().numpy())

                loss.backward()
                self.optimizer.step()
                self.tensorboard.add_scalar("train_loss", loss.item())
                self.tensorboard.add_scalar("train_score", score)

                if idx % CHECKPOINT_STEP == CHECKPOINT_STEP - 1:
                    save_checkpoint(self.net, {"epoch": epoch}, "{}-{}".format(epoch, self.config["CHECKPOINT"]))

            save_checkpoint(self.net, {"epoch": epoch}, "{}-{}".format(epoch, self.config["CHECKPOINT"]))
            print(">>>Trainer epoch finished")
        print(">> Completed")
