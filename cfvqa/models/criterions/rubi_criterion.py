import torch.nn as nn
import torch
import torch.nn.functional as F
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options

class RUBiCriterion(nn.Module):

    def __init__(self, question_loss_weight=1.0):
        super().__init__()

        Logger()(f'RUBiCriterion, with question_loss_weight = ({question_loss_weight})')

        self.question_loss_weight = question_loss_weight
        self.fusion_loss = nn.CrossEntropyLoss()
        self.question_loss = nn.CrossEntropyLoss()
     
    def forward(self, net_out, batch):
        out = {}
        # logits = net_out['logits']
        logits_q = net_out['logits_q']
        logits_rubi = net_out['logits_all']
        class_id = batch['class_id'].squeeze(1)
        fusion_loss = self.fusion_loss(logits_rubi, class_id)
        question_loss = self.question_loss(logits_q, class_id)
        loss = fusion_loss + self.question_loss_weight * question_loss

        out['loss'] = loss
        out['loss_mm_q'] = fusion_loss
        out['loss_q'] = question_loss
        return out
