from copy import deepcopy
import itertools
import os
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
import block
from block.models.networks.vqa_net import factory_text_enc
from block.models.networks.mlp import MLP

from .utils import mask_softmax

from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable

class SANNet(nn.Module):

    def __init__(self,
            txt_enc={},
            self_q_att=False,
            agg={},
            classif={},
            wid_to_word={},
            word_to_wid={},
            aid_to_ans=[],
            ans_to_aid={},
            fusion={},
            residual=False,
            q_single=False
            ):
        super().__init__()
        self.self_q_att = self_q_att
        self.agg = agg
        assert self.agg['type'] in ['max', 'mean']
        self.classif = classif
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        self.fusion = fusion
        self.residual = residual
        
        # Modules
        self.txt_enc = self.get_text_enc(self.wid_to_word, txt_enc)
        if self.self_q_att:
            self.q_att_linear0 = nn.Linear(2400, 512)
            self.q_att_linear1 = nn.Linear(512, 2)

        if q_single:
            self.txt_enc_single = self.get_text_enc(self.wid_to_word, txt_enc)
            if self.self_q_att:
                self.q_att_linear0_single = nn.Linear(2400, 512)
                self.q_att_linear1_single = nn.Linear(512, 2)

        if self.classif['mlp']['dimensions'][-1] != len(self.aid_to_ans):
            Logger()(f"Warning, the classif_mm output dimension ({self.classif['mlp']['dimensions'][-1]})" 
             f"doesn't match the number of answers ({len(self.aid_to_ans)}). Modifying the output dimension.")
            self.classif['mlp']['dimensions'][-1] = len(self.aid_to_ans) 

        self.classif_module = MLP(**self.classif['mlp'])

        # UpDn
        q_dim = self.fusion['input_dims'][0]
        v_dim = self.fusion['input_dims'][1]
        output_dim = self.fusion['output_dim']
        att_size = 512
        self.v_att = Attention(v_dim, v_dim, att_size, 36, output_dim, drop_ratio=0.5)
        self.txt_enc.rnn = QuestionEmbedding(620, q_dim, 1, False, 0.0)

        self.q_net = FCNet([q_dim, output_dim])
        # self.v_net = FCNet([v_dim, output_dim])

        Logger().log_value('nparams',
            sum(p.numel() for p in self.parameters() if p.requires_grad),
            should_print=True)

        Logger().log_value('nparams_txt_enc',
            self.get_nparams_txt_enc(),
            should_print=True)

      
    def get_text_enc(self, vocab_words, options):
        """
        returns the text encoding network. 
        """
        return factory_text_enc(self.wid_to_word, options)

    def get_nparams_txt_enc(self):
        params = [p.numel() for p in self.txt_enc.parameters() if p.requires_grad]
        if self.self_q_att:
            params += [p.numel() for p in self.q_att_linear0.parameters() if p.requires_grad]
            params += [p.numel() for p in self.q_att_linear1.parameters() if p.requires_grad]
        return sum(params)

    def forward(self, batch):
        v = batch['visual']
        q = batch['question']
        l = batch['lengths'].data
        c = batch['norm_coord']
        nb_regions = batch.get('nb_regions')

        out = {}

        q_emb = self.process_question(q, l,)
        out['v_emb'] = v.mean(1)
        out['q_emb'] = q_emb

        # single txt encoder
        if self.txt_enc_single is not None:
            out['q_emb'] = self.process_question(q, l, self.txt_enc_single, self.q_att_linear0_single, self.q_att_linear1_single)

        # New
        q_repr = self.q_net(q_emb)
        joint_repr = self.v_att(q_repr, v)

        logits = self.classif_module(joint_repr)
        out['logits'] = logits

        return out

    def process_question(self, q, l, txt_enc=None, q_att_linear0=None, q_att_linear1=None):
        if txt_enc is None:
            txt_enc = self.txt_enc
        q_emb = txt_enc.embedding(q)
        q = txt_enc.rnn(q_emb)
        return q

    def process_answers(self, out, key=''):
        batch_size = out[f'logits{key}'].shape[0]
        _, pred = out[f'logits{key}'].data.max(1)
        pred.squeeze_()
        if batch_size != 1:
            out[f'answers{key}'] = [self.aid_to_ans[pred[i].item()] for i in range(batch_size)]
            out[f'answer_ids{key}'] = [pred[i].item() for i in range(batch_size)]
        else:
            out[f'answers{key}'] = [self.aid_to_ans[pred.item()]]
            out[f'answer_ids{key}'] = [pred.item()]
        return out

class Attention(nn.Module): # Extend PyTorch's Module class
    def __init__(self, v_dim, q_dim, att_size, img_seq_size, output_size, drop_ratio):
        super(Attention, self).__init__() # Must call super __init__()
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.att_size = att_size
        self.img_seq_size = img_seq_size
        self.output_size = output_size
        self.drop_ratio = drop_ratio

        self.tan = nn.Tanh()
        self.dp = nn.Dropout(drop_ratio)
        self.sf = nn.Softmax()

        self.fc11 = nn.Linear(q_dim, 768, bias=True)
        # self.fc111 = nn.Linear(768, 640, bias=True)
        self.fc111 = nn.Linear(768, att_size, bias=True)
        self.fc12 = nn.Linear(v_dim, 768, bias=False)
        # self.fc121 = nn.Linear(768, 640, bias=False)
        self.fc121 = nn.Linear(768, att_size, bias=False)
        self.linear_second = nn.Linear(att_size, att_size, bias=False)
        # self.linear_second = nn.Linear(att_size, img_seq_size, bias=False)
        self.fc13 = nn.Linear(att_size, 1, bias=True)

        self.fc21 = nn.Linear(q_dim, att_size, bias=True)
        self.fc22 = nn.Linear(v_dim, att_size, bias=False)
        self.fc23 = nn.Linear(att_size, 1, bias=True)

        self.fc = nn.Linear(v_dim, output_size, bias=True)

        # d = input_size | m = img_seq_size | k = att_size
    def forward(self, ques_feat, img_feat):  # ques_feat -- [batch, d] | img_feat -- [batch_size, m, d]
        # print(img_feat.size(), ques_feat.size())
        # print(self.v_dim, self.q_dim)
        # print("=======================================================================") 
        B = ques_feat.size(0)

        # Stack 1
        
        ques_emb_1 = self.fc11(ques_feat) 
        ques_emb_1 = self.fc111(ques_emb_1) # [batch_size, att_size]
        img_emb_1 = self.fc12(img_feat)
        img_emb_1 = self.fc121(img_emb_1)

        # print(ques_emb_1.size(), img_emb_1.size())
        # print("=======================================================================") 
       
        # h1 = self.tan(ques_emb_1.view(B, 1, self.att_size) + img_emb_1)
        h1 = self.tan(ques_emb_1.view(B, 1, self.att_size) + img_emb_1)
        h1_emb = self.linear_second(h1) 
        h1_emb = self.fc13(h1_emb)
        
        p1 = self.sf(h1_emb.view(-1, self.img_seq_size)).view(B, 1, self.img_seq_size)

        # Weighted sum
        img_att1 = p1.matmul(img_feat)
        u1 = ques_feat + img_att1.view(-1, self.v_dim)

        # Stack 2
        ques_emb_2 = self.fc21(u1)  # [batch_size, att_size]
        img_emb_2 = self.fc22(img_feat)

        h2 = self.tan(ques_emb_2.view(B, 1, self.att_size) + img_emb_2)

        h2_emb = self.fc23(self.dp(h2))
        p2 = self.sf(h2_emb.view(-1, self.img_seq_size)).view(B, 1, self.img_seq_size)

        # Weighted sum
        img_att2 = p2.matmul(img_feat)
        u2 = u1 + img_att2.view(-1, self.v_dim)

        return u2

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

        
class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output
