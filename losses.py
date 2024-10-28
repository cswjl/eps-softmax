import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

    
class ECELoss(nn.Module): # Eps-Softmax with CE loss (ECE)
    def __init__(self, m, eps=1e-8):
        super(ECELoss, self).__init__()
        self.m = m
        # eps can affect the effect of this loss
        self.eps = eps
    def forward(self, input, target):
        input = F.softmax(input, dim=1).clone()
        input[torch.arange(len(input)), torch.argmax(input, dim=1)] += self.m
        input = input / (self.m + 1)
        input = torch.clamp(input, min=self.eps)
        log_soft_out = torch.log(input)
        loss = F.nll_loss(log_soft_out, target)
        return loss.mean()

class ECEandMAE(nn.Module):
    def __init__(self, m, alpha=1, beta=1, eps=1e-8):
        super(ECEandMAE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ece = ECELoss(m, eps=eps)
        self.mae = MAELoss()
        
    def forward(self, input, target):
        loss = self.alpha * self.ece(input, target) + self.beta * self.mae(input, target)
        return loss
class EFLandMAE(nn.Module):
    def __init__(self, m, alpha=1, beta=1, gamma=0.1, eps=1e-8):
        super(EFLandMAE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.efl = EFocalLoss(m=m, gamma=gamma, eps=eps)
        self.mae = MAELoss()
        
    def forward(self, input, target):
        loss = self.alpha * self.efl(input, target) + self.beta * self.mae(input, target)
        return loss    


class EFocalLoss(nn.Module): # Eps-Softmax with FL loss (EFL)
    def __init__(self, gamma=0.1, alpha=None, size_average=True, m=0, eps=1e-8):
        super(EFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.m = m
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        # eps affects the effect of this loss
        self.eps = eps
    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        input = F.softmax(input, dim=1).clone()
        input[torch.arange(len(input)), torch.argmax(input, dim=1)] += self.m
        input = input / (self.m + 1)
        input = torch.clamp(input, min=self.eps)

        logpt = torch.log(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
        
class MSELoss(nn.Module): # with softmax
    def __init__(self):
        super(MSELoss, self).__init__()
    def forward(self, input, target):
        input = F.softmax(input, dim=1)
        label_one_hot = F.one_hot(target, input.shape[1]).float().to(input.device)
        loss = (input - label_one_hot)**2
        return loss.mean()


class SCELoss(nn.Module):
    def __init__(self, a=1, b=1):
        super(SCELoss, self).__init__()
        self.a = a
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-8, max=1.0)
        label_one_hot = F.one_hot(labels, pred.shape[1]).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.a * ce + self.b * rce.mean()
        return loss


class RCELoss(nn.Module):
    def __init__(self, scale=1.0):
        super(RCELoss, self).__init__()
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-8, max=1.0)
        label_one_hot = F.one_hot(labels, pred.shape[1]).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        loss = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * loss.mean()


class NCELoss(nn.Module):
    def __init__(self, scale=1.0):
        super(NCELoss, self).__init__()
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, pred.shape[1]).float().to(pred.device)
        loss = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        return self.scale * loss.mean()

class MAELoss(nn.Module):
    def __init__(self, scale=2.0):
        super(MAELoss, self).__init__()
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, pred.shape[1]).float().to(pred.device)
        loss = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * loss.mean()

class GCELoss(nn.Module):
    def __init__(self, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-8, max=1.0)
        label_one_hot = F.one_hot(labels, pred.shape[1]).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()


class AGCELoss(nn.Module):
    def __init__(self, a=1, q=2, scale=1.):
        super(AGCELoss, self).__init__()
        self.a = a
        self.q = q
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, pred.shape[1]).float().to(pred.device)
        loss = ((self.a+1)**self.q - torch.pow(self.a + torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean() * self.scale

class AUELoss(nn.Module):
    def __init__(self, num_classes=10, a=1.5, q=0.9, scale=1.0):
        super(AUELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.q = q
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (torch.pow(self.a - torch.sum(label_one_hot * pred, dim=1), self.q) - (self.a-1)**self.q)/ self.q
        return loss.mean() * self.scale


class AExpLoss(torch.nn.Module):
    def __init__(self, a=3, scale=1.0):
        super(AExpLoss, self).__init__()
        self.a = a
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, pred.shape[1]).float().to(pred.device)
        loss = torch.exp(-torch.sum(label_one_hot * pred, dim=1) / self.a)
        return loss.mean() * self.scale

class NCEandRCE(nn.Module):
    def __init__(self, alpha=1., beta=1.):
        super(NCEandRCE, self).__init__()
        self.nce = NCELoss(scale=alpha)
        self.rce = RCELoss(scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)

class NCEandMAE(nn.Module):
    def __init__(self, alpha=1., beta=1.):
        super(NCEandMAE, self).__init__()
        self.nce = NCELoss(scale=alpha)
        self.mae = MAELoss(scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.mae(pred, labels)

class NLNL(torch.nn.Module):
    def __init__(self, train_loader, num_classes=10, ln_neg=1):
        super(NLNL, self).__init__()
        self.num_classes = num_classes
        self.ln_neg = ln_neg
        weight = torch.FloatTensor(num_classes).zero_() + 1.
        if not hasattr(train_loader.dataset, 'targets'):
            weight = [1] * num_classes
            weight = torch.FloatTensor(weight)
        else:
            for i in range(num_classes):
                weight[i] = (torch.from_numpy(np.array(train_loader.dataset.targets)) == i).sum()
            weight = 1 / (weight / weight.max())
        self.weight = weight.cuda()
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.criterion_nll = torch.nn.NLLLoss()

    def forward(self, pred, labels):
        labels_neg = (labels.unsqueeze(-1).repeat(1, self.ln_neg)
                      + torch.LongTensor(len(labels), self.ln_neg).cuda().random_(1, self.num_classes)) % self.num_classes
        labels_neg = torch.autograd.Variable(labels_neg)

        assert labels_neg.max() <= self.num_classes-1
        assert labels_neg.min() >= 0
        assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(labels)*self.ln_neg

        s_neg = torch.log(torch.clamp(1. - F.softmax(pred, 1), min=1e-5, max=1.))
        s_neg *= self.weight[labels].unsqueeze(-1).expand(s_neg.size()).cuda()
        labels = labels * 0 - 100
        loss = self.criterion(pred, labels) * float((labels >= 0).sum())
        loss_neg = self.criterion_nll(s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
        loss = ((loss+loss_neg) / (float((labels >= 0).sum())+float((labels_neg[:, 0] >= 0).sum())))
        return loss

class FocalLoss(torch.nn.Module):
    '''
        https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    '''

    def __init__(self, gamma=0.5, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, gamma=0.5, size_average=True, scale=1.0):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = self.scale * loss / normalizor

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class NFLandRCE(torch.nn.Module):
    def __init__(self, alpha=1., beta=1., gamma=0.5):
        super(NFLandRCE, self).__init__()
        self.nfl = NormalizedFocalLoss(gamma=gamma, scale=alpha)
        self.rce = RCELoss(scale=beta)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.rce(pred, labels)


class NFLandMAE(torch.nn.Module):
    def __init__(self, alpha=1., beta=1., gamma=0.5):
        super(NFLandMAE, self).__init__()
        self.nfl = NormalizedFocalLoss(gamma=gamma, scale=alpha)
        self.mae = MAELoss(scale=beta)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.mae(pred, labels)


class NCEandAGCE(torch.nn.Module):
    def __init__(self, alpha=1., beta = 1., a=3, q=1.5):
        super(NCEandAGCE, self).__init__()
        self.nce = NCELoss(scale=alpha)
        self.agce = AGCELoss(a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.agce(pred, labels)


class NCEandAUE(torch.nn.Module):
    def __init__(self, alpha=1., beta=1., a=6, q=1.5):
        super(NCEandAUE, self).__init__()
        self.nce = NCELoss(scale=alpha)
        self.aue = AUELoss(a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.aue(pred, labels)

class NCEandAEL(torch.nn.Module):
    def __init__(self, alpha=1., beta=4., a=2.5):
        super(NCEandAEL, self).__init__()
        self.nce = NCELoss(scale=alpha)
        self.aue = AExpLoss(a=a, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.aue(pred, labels)


class NFLandAGCE(torch.nn.Module):
    def __init__(self, alpha=1., beta = 1., a=3, q=2):
        super(NFLandAGCE, self).__init__()
        self.nce = NormalizedFocalLoss(scale=alpha)
        self.agce = AGCELoss(a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.agce(pred, labels)


class NFLandAUE(torch.nn.Module):
    def __init__(self, alpha=1., beta = 1., a=1.5, q=0.9):
        super(NFLandAUE, self).__init__()
        self.nce = NormalizedFocalLoss(scale=alpha)
        self.aue = AUELoss(a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.aue(pred, labels)


class NFLandAEL(torch.nn.Module):
    def __init__(self, alpha=1., beta = 1., a=3):
        super(NFLandAEL, self).__init__()
        self.nce = NormalizedFocalLoss(scale=alpha)
        self.ael = AExpLoss(a=a, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.ael(pred, labels)


class NormalizedNegativeFocalLoss(torch.nn.Module):
    def __init__(self, gamma=0.5, min_prob=1e-8) -> None:
        super().__init__()
        self.gamma = gamma
        self.min_prob = min_prob
        self.logmp = torch.tensor(self.min_prob).log()
        self.A = - (1 - min_prob)**gamma * self.logmp
    
    def forward(self, input, target):
        logmp = self.logmp.to(input.device)
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1).clamp(min=logmp)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = 1 - (self.A - loss) / (input.shape[1] * self.A - normalizor)
        return loss.mean()
    

class NormalizedNegativeCrossEntropy(torch.nn.Module):
    def __init__(self, min_prob=1e-8) -> None:
        super().__init__()
        self.min_prob = min_prob
        self.A = - torch.tensor(min_prob).log()
    
    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = pred.clamp(min=self.min_prob, max=1)
        pred = self.A + pred.log() # - log(eps) - (- log(p(k|x)))
        label_one_hot = F.one_hot(labels, pred.shape[1]).to(pred.device)
        nnce = 1 - (label_one_hot * pred).sum(dim=1) / pred.sum(dim=1)
        return nnce.mean()
    
class NCEandNNCE(torch.nn.Module):
    def __init__(self, alpha=1., beta = 1., eps=1e-8):
        super(NCEandNNCE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nce = NCELoss(scale=1)
        self.nnce = NormalizedNegativeCrossEntropy(min_prob=eps)

    def forward(self, pred, labels):
        return self.alpha * self.nce(pred, labels) + self.beta * self.nnce(pred, labels)
    
class NFLandNNFL(torch.nn.Module):
    def __init__(self, alpha=1., beta = 1., eps=1e-8):
        super(NFLandNNFL, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nfl = NormalizedFocalLoss(scale=1)
        self.nnfl = NormalizedNegativeFocalLoss(min_prob=eps, gamma=0.5)

    def forward(self, pred, labels):
        return self.alpha * self.nfl(pred, labels) + self.beta * self.nnfl(pred, labels)


class CEandLC(nn.Module):
    def __init__(self, delta):
        super(CEandLC, self).__init__()
        self.delta = delta
    def forward(self, input, target):
        temp = 1/self.delta
        norms = torch.norm(input, p=2, dim=-1, keepdim=True) + 1e-8
        logits_norm = torch.div(input, norms) * self.delta
        clip = (norms > temp).expand(-1, input.shape[-1])
        logits_final = torch.where(clip, logits_norm, input)
        input = F.softmax(logits_final, dim=1)
        input = torch.clamp(input, min=1e-12, max=1.0)
        log_soft_out = torch.log(input)
        loss = F.nll_loss(log_soft_out, target)
        return loss.mean()

def get_diff_logits(y_pred, y_true):
    y_true_logits = torch.sum( y_pred * y_true, dim=1, keepdim=True)
    return y_pred - y_true_logits

class LDRLoss_V1(nn.Module):
    def __init__(self, threshold=0.1, Lambda=1.0):
        super(LDRLoss_V1, self).__init__()
        self.threshold = threshold
        self.Lambda = Lambda

    def forward(self, y_pred, y_true):
        num_class = y_pred.shape[1]
        y_true = F.one_hot(y_true, num_class).float()
        y_pred = torch.nn.functional.softplus(y_pred)
        y_denorm = torch.mean(y_pred, dim=1, keepdim=True)
        y_pred = y_pred/y_denorm
        diff_logits = self.threshold*(1-y_true) + get_diff_logits(y_pred, y_true)
        diff_logits = diff_logits/self.Lambda
        max_diff = torch.max(diff_logits, dim=1, keepdim=True).values.detach()
        diff_logits = diff_logits - max_diff
        diff_logits = torch.exp(diff_logits)
        loss = self.Lambda*(torch.log(torch.mean(diff_logits, dim=1, keepdim=True)) + max_diff)
        return loss.mean()