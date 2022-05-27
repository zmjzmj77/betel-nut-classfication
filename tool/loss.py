import torch.nn.functional as F
import torch
import torch.nn as nn

def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.2, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)

def mix_loss(loss_func, out, label, label1, lam):
    return lam * loss_func(out, label) + (1 - lam) * loss_func(out, label1)

if __name__ == '__main__':
    input = torch.FloatTensor([[1,2,3],
                          [4,2,3]])
    target = torch.tensor([1,2])

    loss_f = LabelSmoothingCrossEntropy()
    loss = loss_f(input, target)
    print(loss)