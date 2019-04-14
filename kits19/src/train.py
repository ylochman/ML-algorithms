import torch
import torch.nn as nn
import torch.utils.data
import tqdm
import torch.functional as F
import numpy as np

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
        self.loss_values = []

    def run(self, dataloader, epochs=1):
        print(">> Running trainer")
        for epoch in range(epochs):
            print(">>> Epoch %s" % epoch)
            for idx, (image, target) in enumerate(tqdm.tqdm(dataloader, ascii=True)):
                image, target = image.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                predict = self.net(image)
                loss = self.loss(predict, target)
                loss.backward()
                self.optimizer.step()
                self.loss_values.append(loss.item())
                print(">>> Last 10 batches loss: {}".format(np.mean(self.loss_values[-10:])))
            save_checkpoint(self.net, {"epoch": epoch}, "{}-{}".format(epoch, self.config["CHECKPOINT"]))
            print(">>>Trainer epoch finished")
        print(">> Completed")
