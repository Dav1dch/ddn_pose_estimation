import torch
from torch import nn
import torch.nn.functional as F


def vdot(v1, v2):
    """
    Dot product along the dim=1
    :param v1: N x d
    :param v2: N x d
    :return: N x 1
    """
    out = torch.mul(v1, v2)
    out = torch.sum(out, 1)
    return out


class QuaternionLoss(nn.Module):
    """
    Implements distance between quaternions as mentioned in
    D. Huynh. Metrics for 3D rotations: Comparison and analysis
    """

    def __init__(self):
        super(QuaternionLoss, self).__init__()

    def forward(self, q1, q2):
        """
        :param q1: N x 4
        :param q2: N x 4
        :return:
        """
        loss = 1 - torch.pow(vdot(q1, q2), 2)
        loss = torch.mean(loss)
        return loss


class PoseLoss(nn.Module):
    def __init__(self, sx=0.0, sq=0.0, learn_beta=False):
        super(PoseLoss, self).__init__()
        self.learn_beta = learn_beta
        self.quaternionLoss = QuaternionLoss()

        if not self.learn_beta:
            self.sx = 0.0
            self.sq = -6.25

        self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=self.learn_beta)
        self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=self.learn_beta)
        self.loss_print = None

    def forward(self, pred_x, pred_q, target_x, target_q):
        # pred_q = F.normalize(pred_q, p=2, dim=1)
        loss_x = F.l1_loss(pred_x, target_x)
        # loss_q = F.l1_loss(pred_q, target_q)
        loss_q = self.quaternionLoss(pred_q, target_q)

        loss = (
            torch.exp(-self.sx) * loss_x
            + self.sx
            + torch.exp(-self.sq) * loss_q
            + self.sq
        )

        self.loss_print = [loss.item(), loss_x.item(), loss_q.item()]

        return loss, loss_x.item(), loss_q.item()
