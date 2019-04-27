import torch


def dice(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return ((2. * intersection + smooth) /
            (iflat.sum() + tflat.sum() + smooth))


class Dice_loss(torch.nn.Module):
    def init(self):
        super(Dice_loss, self).init()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_().requires_grad_(True)
        else:
            s = torch.FloatTensor(1).zero_().requires_grad_(True)
        for i, c in enumerate(zip(input, target)):
            s = s + (1 - dice(c[0], c[1]))
        return s / (i + 1)
