import sys
import copy
import torch
import torch.nn as nn
import os
import json
from bootstrap.lib.options import Options
from bootstrap.models.networks.data_parallel import DataParallel
from block.models.networks.vqa_net import VQANet as AttentionNet
from bootstrap.lib.logger import Logger

from .rubi import RUBiNet
from .cfvqa import CFVQA

def factory(engine):
    mode = list(engine.dataset.keys())[0]
    dataset = engine.dataset[mode]
    opt = Options()['model.network']


    if opt['base'] == 'smrl':
        from .smrl_net import SMRLNet as BaselineNet
    elif opt['base'] == 'updn':
        from .updn_net import UpDnNet as BaselineNet
    elif opt['base'] == 'san':
        from .san_net import SANNet as BaselineNet
    else:
        raise ValueError(opt['base'])

    orig_net = BaselineNet(
        txt_enc=opt['txt_enc'],
        self_q_att=opt['self_q_att'],
        agg=opt['agg'],
        classif=opt['classif'],
        wid_to_word=dataset.wid_to_word,
        word_to_wid=dataset.word_to_wid,
        aid_to_ans=dataset.aid_to_ans,
        ans_to_aid=dataset.ans_to_aid,
        fusion=opt['fusion'],
        residual=opt['residual'],
        q_single=opt['q_single'],
    )

    if opt['name'] == 'baseline':
        net = orig_net

    elif opt['name'] == 'rubi':
        net = RUBiNet(
            model=orig_net,
            output_size=len(dataset.aid_to_ans),
            classif=opt['rubi_params']['mlp_q']
        )

    elif opt['name'] == 'cfvqa':
        net = CFVQA(
            model=orig_net,
            output_size=len(dataset.aid_to_ans),
            classif_q=opt['cfvqa_params']['mlp_q'],
            classif_v=opt['cfvqa_params']['mlp_v'],
            fusion_mode=opt['fusion_mode'],
            is_va=True
        )

    elif opt['name'] == 'cfvqasimple':
        net = CFVQA(
            model=orig_net,
            output_size=len(dataset.aid_to_ans),
            classif_q=opt['cfvqa_params']['mlp_q'],
            classif_v=None,
            fusion_mode=opt['fusion_mode'],
            is_va=False
        )

    else:
        raise ValueError(opt['name'])

    if Options()['misc.cuda'] and torch.cuda.device_count() > 1:
        net = DataParallel(net)

    return net

