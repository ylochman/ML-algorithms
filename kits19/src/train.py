import torch
import torch.nn as nn
import torch.utils.data
import tqdm
import torch.functional as F
import numpy as np
from tensorboardX import SummaryWriter

from src.score import score_function
from src.utils import save_checkpoint


class Trainer:
    def __init__(self, net, config):
        net.train()
        net.to(config['DEVICE'])
        self.device = config['DEVICE']
        self.net = net
        self.config = config
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=config['LR'])
        self.tensorboard = SummaryWriter()

    def run(self, dataloader, epochs=1):
        print(">> Running trainer")
        for epoch in range(epochs):
            print(">>> Epoch %s" % epoch)
            for idx, (image, target) in enumerate(tqdm.tqdm(dataloader, ascii=True)):
                image, target = image.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                predict = self.net(image)
                loss = self.loss(predict, target)
                # todo Too slow. Increases training time from 2h -> 3.5h. Either move it to evaluation only or improve speed
                # batch_prediction = torch.argmax(torch.softmax(predict.cpu(), 1), 1, keepdim=True)
                # score = score_function(batch_prediction, target.cpu())
                loss.backward()
                self.optimizer.step()
                self.tensorboard.add_scalar("train_loss", loss.item())
                # self.tensorboard.add_scalar("train_score", score)
            save_checkpoint(self.net, {"epoch": epoch}, "{}-{}".format(epoch, self.config["CHECKPOINT"]))
            print(">>>Trainer epoch finished")
        print(">> Completed")
