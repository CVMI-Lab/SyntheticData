import os

import torch
import pickle
from tqdm import tqdm
import math
import torch.nn.functional as F
import numpy as np


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster

def cosine_lr_const_warmup(optimizer, base_lrs, warmup_length, steps, warmup_const=1e-5):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = warmup_const
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster

def get_cosine_lr_value(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def torch_save(classifier, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(classifier.cpu(), f)


def torch_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, 'predict_proba'):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        # import ipdb
        # ipdb.set_trace(context=20)
        logprobs = F.log_softmax(x, dim=-1) # B C

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)) # B 1
        nll_loss = nll_loss.squeeze(1)                                  # B
        smooth_loss = -logprobs.mean(dim=-1)                            # B
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class SoftTargetCrossEntropy_T(torch.nn.Module):
    '''
    from timm, abandon
    '''

    def __init__(self, T):
        super(SoftTargetCrossEntropy_T, self).__init__()
        self.T = T

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: student logit
            target: teacher logit

        Returns:

        '''
        soft_labels = torch.softmax(target/self.T, dim=1)
        loss = torch.sum(-soft_labels * F.log_softmax(x, dim=-1), dim=-1)
        # loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

class SoftTargetCrossEntropy(torch.nn.Module):
    '''
    from timm
    '''

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: student logit
            target: teacher logit

        Returns:

        '''
        # soft_labels = torch.softmax(target/self.T, dim=1)
        # loss = torch.sum(-soft_labels * F.log_softmax(x, dim=-1), dim=-1)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

class KD_loss(torch.nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(KD_loss, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class Hard_pseudo_label_CE(torch.nn.Module):
    '''
    from timm
    '''

    def __init__(self):
        super(Hard_pseudo_label_CE, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: student logit
            target: teacher logit

        Returns:

        '''
        hard_pseudo_labels = torch.argmax(torch.softmax(target, dim=1), dim=1)
        loss = self.loss_fn(x, hard_pseudo_labels)
        # loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss

def onehot(labels: torch.Tensor, label_num):
    return torch.zeros(labels.shape[0], label_num, device=labels.device).scatter_(1, labels.view(-1, 1), 1)

class Poly1_cross_entropy(torch.nn.Module):
    '''
        https://arxiv.org/pdf/2204.12511.pdf
    '''

    def __init__(self, epsilon=1.0):
        super(Poly1_cross_entropy, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.epsilon=epsilon

    def forward(self, logits, labels):
        # pt, CE, and Poly1 have shape [batch].
        # import ipdb
        # ipdb.set_trace(context=20)
        onehot_labels = onehot(labels, logits.shape[-1])
        pt = torch.sum(onehot_labels * F.softmax(logits, dim=1), dim=-1)
        # CE = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        CE = self.loss_fn(logits, labels)
        Poly1 = CE + self.epsilon * (1 - pt)
        return Poly1.mean()