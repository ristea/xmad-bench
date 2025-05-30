import torch

bce = torch.nn.BCELoss()
ce = torch.nn.CrossEntropyLoss()


def bce_loss(predictions, labels):

    loss = bce(predictions, labels)
    return loss


def ce_loss(predictions, labels):
    loss1 = ce(predictions[:, :2], labels[:, 0].long())
    loss2 = ce(predictions[:, 2:], labels[:, 1].long())
    return loss1 + loss2
