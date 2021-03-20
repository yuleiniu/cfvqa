import torch
import torch.nn as nn
from block.models.networks.mlp import MLP
from .utils import grad_mul_const # mask_softmax, grad_reverse, grad_reverse_mask, 

eps = 1e-12

class CFVQASimple(nn.Module):
    """
    Wraps another model
    The original model must return a dictionnary containing the 'logits' key (predictions before softmax)
    Returns:
        - logits_vq: the original predictions of the model, i.e., NIE
        - logits_q: the predictions from the question-only branch
        - logits_all: the predictions from the ensemble model
        - logits_cfvqa: the predictions based on CF-VQA, i.e., TIE
    => Use `logits_all` and `logits_q` for the loss
    """
    def __init__(self, model, output_size, classif_q, fusion_mode, end_classif=True):
        super().__init__()
        self.net = model
        self.end_classif = end_classif

        # Q branch
        self.q_1 = MLP(**classif_q)
        if self.end_classif: # default: True (following RUBi)
            self.q_2 = nn.Linear(output_size, output_size)

        self.fusion_mode = fusion_mode
        self.constant = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        out = {}
        # model prediction
        net_out = self.net(batch)
        logits = net_out['logits']

        # Q branch
        q_embedding = net_out['q_emb']  # N * q_emb
        q_embedding = grad_mul_const(q_embedding, 0.0) # don't backpropagate
        q_pred = self.q_1(q_embedding)

        # both q and k are the facts
        z_qk = self.fusion(logits, q_pred, q_fact=True,  k_fact=True) # te
        # q is the fact while k is the counterfactuals
        z_q = self.fusion(logits, q_pred, q_fact=True,  k_fact=False) # nie
        
        logits_cfvqa = z_qk - z_q

        if self.end_classif:
            q_out = self.q_2(q_pred)
        else:
            q_out = q_pred

        out['logits_all'] = z_qkv # for optimization
        out['logits_vq']  = logits # predictions of the original VQ branch, i.e., NIE
        out['logits_cfvqa'] = logits_cfvqa # predictions of CFVQA, i.e., TIE
        out['logits_q'] = q_out # for optimization

        out['z_nde'] = self.fusion(logits.clone().detach(), q_pred.clone().detach(), v_pred.clone().detach(), q_fact=True,  k_fact=False, v_fact=False) # tiekv
        return out

    def process_answers(self, out, key=''):
        out = self.net.process_answers(out, key='_all')
        out = self.net.process_answers(out, key='_vq')
        out = self.net.process_answers(out, key='_cfvqa')
        out = self.net.process_answers(out, key='_q')
        return out

    def fusion(self, z_k, z_q, z_v, q_fact=False, k_fact=False, v_fact=False):

        z_k, z_q, z_v = self.transform(z_k, z_q, z_v, q_fact, k_fact, v_fact)

        if self.fusion_mode == 'rubi':
            z = z_k * (torch.sigmoid(z_q) + torch.sigmoid(z_v))

        elif self.fusion_mode == 'harmonic':
            p = z_k * z_q * z_v
            z = torch.log(p + eps) - torch.log1p(p)

        elif self.fusion_mode == 'sum':
            z = torch.log(torch.sigmoid(z_k + z_q + z_v) + eps)

        return z

    def transform(self, z_k, z_q, z_v, q_fact=False, k_fact=False, v_fact=False):  

        if not k_fact:
            z_k = self.constant * torch.ones_like(z_k).cuda()

        if not q_fact:
            z_q = self.constant * torch.ones_like(z_q).cuda()

        if not v_fact:
            z_v = self.constant * torch.ones_like(z_v).cuda()

        if self.mode == 'harmonic':
            z_k = torch.sigmoid(z_k)
            z_q = torch.sigmoid(z_q)
            z_v = torch.sigmoid(z_v)

        return z_k, z_q, z_v