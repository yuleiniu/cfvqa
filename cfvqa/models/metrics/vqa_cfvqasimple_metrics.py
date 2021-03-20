import torch
import torch.nn as nn
import os
import json
from scipy import stats
import numpy as np
from collections import defaultdict

from bootstrap.models.metrics.accuracy import accuracy
from block.models.metrics.vqa_accuracies import VQAAccuracies
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

class VQAAccuracy(nn.Module):

    def __init__(self, topk=[1,5]):
        super().__init__()
        self.topk = topk
        self.metric_list = ['_all', '_vq', '_cfvqa', '_q']

    def forward(self, cri_out, net_out, batch):
        out = {}
        class_id = batch['class_id'].data.cpu()
        for key in self.metric_list:
            logits = net_out[f'logits{key}'].data.cpu()
            acc_out = accuracy(logits, class_id, topk=self.topk)
            for i, k in enumerate(self.topk):
                out[f'accuracy{key}_top{k}'] = acc_out[i]
        return out


class VQACFVQASimpleMetrics(VQAAccuracies):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_list = ['_all', '_vq', '_cfvqa', '_q']
        if Options()['dataset.eval_split'] == 'test': # 0430
            self.accuracy = None
        else:
            self.accuracy = VQAAccuracy()
        self.rm_dir_rslt = 1 if Options()['dataset.train_split'] is not None else 0

    def forward(self, cri_out, net_out, batch):
        out = {}
        if self.accuracy is not None:
            out = self.accuracy(cri_out, net_out, batch)

        # add answers and answer_ids keys to net_out
        net_out = self.engine.model.network.process_answers(net_out)

        batch_size = len(batch['index'])
        for i in range(batch_size):
            
            # Open Ended Accuracy (VQA-VQA2)
            if self.open_ended:
                for key in self.metric_list:
                    pred_item = {
                        'question_id': batch['question_id'][i],
                        'answer': net_out[f'answers{key}'][i]
                    }
                    self.results[key].append(pred_item)

                # if self.dataset.split == 'test': # 0430
                #     pred_item = {
                #         'question_id': batch['question_id'][i],
                #         'answer': net_out[f'answers{key}'][i]
                #         # 'answer': net_out[f'answers'][i]
                #     }
                #     # if 'is_testdev' in batch and batch['is_testdev'][i]: # 0430
                #     #     self.results_testdev.append(pred_item)

                #     if self.logits['tensor'] is None:
                #         self.logits['tensor'] = torch.FloatTensor(len(self.dataset), logits.size(1))

                #     self.logits['tensor'][self.idx] = logits[i]
                #     self.logits['qid_to_idx'][batch['question_id'][i]] = self.idx
                    
                #     self.idx += 1

                # TDIUC metrics
                if self.tdiuc:
                    gt_aid = batch['answer_id'][i]
                    gt_ans = batch['answer'][i]
                    gt_type = batch['question_type'][i]
                    self.gt_types.append(gt_type)
                    if gt_ans in self.ans_to_aid:
                        self.gt_aids.append(gt_aid)
                    else:
                        self.gt_aids.append(-1)
                        self.gt_aid_not_found += 1

                    for key in self.metric_list:
                        qid = batch['question_id'][i]
                        pred_aid = net_out[f'answer_ids{key}'][i]
                        self.pred_aids[key].append(pred_aid)

                        self.res_by_type[key][gt_type+'_pred'].append(pred_aid)

                        if gt_ans in self.ans_to_aid:
                            self.res_by_type[key][gt_type+'_gt'].append(gt_aid)
                            if gt_aid == pred_aid:
                                self.res_by_type[key][gt_type+'_t'].append(pred_aid)
                            else:
                                self.res_by_type[key][gt_type+'_f'].append(pred_aid)
                        else:
                            self.res_by_type[key][gt_type+'_gt'].append(-1)
                            self.res_by_type[key][gt_type+'_f'].append(pred_aid)
        return out

    def reset_oe(self):
        self.results = dict()
        self.dir_rslt = dict()
        self.path_rslt = dict()
        for key in self.metric_list:
            self.results[key] = []
            self.dir_rslt[key] = os.path.join(
                self.dir_exp,
                f'results{key}',
                self.dataset.split,
                'epoch,{}'.format(self.engine.epoch))
            os.system('mkdir -p '+self.dir_rslt[key])
            self.path_rslt[key] = os.path.join(
                self.dir_rslt[key],
                'OpenEnded_mscoco_{}_model_results.json'.format(
                    self.dataset.get_subtype()))

            if self.dataset.split == 'test':
                pass
                # self.results_testdev = []
                # self.path_rslt_testdev = os.path.join(
                #     self.dir_rslt,
                #     'OpenEnded_mscoco_{}_model_results.json'.format(
                #         self.dataset.get_subtype(testdev=True)))

                # self.path_logits = os.path.join(self.dir_rslt, 'logits.pth')
                # os.system('mkdir -p '+os.path.dirname(self.path_logits))

                # self.logits = {}
                # self.logits['aid_to_ans'] = self.engine.model.network.aid_to_ans
                # self.logits['qid_to_idx'] = {}
                # self.logits['tensor'] = None

                # self.idx = 0

                # path_aid_to_ans = os.path.join(self.dir_rslt, 'aid_to_ans.json')
                # with open(path_aid_to_ans, 'w') as f:
                #     json.dump(self.engine.model.network.aid_to_ans, f)
    

    def reset_tdiuc(self):
        self.pred_aids = defaultdict(list)
        self.gt_aids = []
        self.gt_types = []
        self.gt_aid_not_found = 0
        self.res_by_type = {key: defaultdict(list) for key in self.metric_list}
    
    
    def compute_oe_accuracy(self):
        logs_name_prefix = Options()['misc'].get('logs_name', '') or ''
        
        for key in self.metric_list:
            logs_name = (logs_name_prefix + key) or "logs"
            with open(self.path_rslt[key], 'w') as f:
                json.dump(self.results[key], f)
            
            # if self.dataset.split == 'test':
            #     with open(self.path_rslt_testdev, 'w') as f:
            #         json.dump(self.results_testdev, f)

            if 'test' not in self.dataset.split:
                call_to_prog = 'python -m block.models.metrics.compute_oe_accuracy '\
                    + '--dir_vqa {} --dir_exp {} --dir_rslt {} --epoch {} --split {} --logs_name {} --rm {} &'\
                    .format(self.dir_vqa, self.dir_exp, self.dir_rslt[key], self.engine.epoch, self.dataset.split, logs_name, self.rm_dir_rslt)
                Logger()('`'+call_to_prog+'`')
                os.system(call_to_prog)


    def compute_tdiuc_metrics(self):
        Logger()('{} of validation answers were not found in ans_to_aid'.format(self.gt_aid_not_found))
        
        for key in self.metric_list:
            Logger()(f'Computing TDIUC metrics for logits{key}')
            accuracy = float(100*np.mean(np.array(self.pred_aids[key])==np.array(self.gt_aids)))
            Logger()('Overall Traditional Accuracy is {:.2f}'.format(accuracy))
            Logger().log_value('{}_epoch.tdiuc.accuracy{}'.format(self.mode, key), accuracy, should_print=False)
            
            types = list(set(self.gt_types))
            sum_acc = []
            eps = 1e-10

            Logger()('---------------------------------------')
            Logger()('Not using per-answer normalization...')
            for tp in types:
                acc = 100*(len(self.res_by_type[key][tp+'_t'])/len(self.res_by_type[key][tp+'_t']+self.res_by_type[key][tp+'_f']))
                sum_acc.append(acc+eps)
                Logger()(f"Accuracy {key} for class '{tp}' is {acc:.2f}")
                Logger().log_value('{}_epoch.tdiuc{}.perQuestionType.{}'.format(self.mode, key, tp), acc, should_print=False)

            acc_mpt_a = float(np.mean(np.array(sum_acc)))
            Logger()('Arithmetic MPT Accuracy {} is {:.2f}'.format(key, acc_mpt_a))
            Logger().log_value('{}_epoch.tdiuc{}.acc_mpt_a'.format(self.mode, key), acc_mpt_a, should_print=False)

            acc_mpt_h = float(stats.hmean(sum_acc))
            Logger()('Harmonic MPT Accuracy {} is {:.2f}'.format(key, acc_mpt_h))
            Logger().log_value('{}_epoch.tdiuc{}.acc_mpt_h'.format(self.mode, key), acc_mpt_h, should_print=False)
            
            Logger()('---------------------------------------')
            Logger()('Using per-answer normalization...')
            for tp in types:
                per_ans_stat = defaultdict(int)
                for g,p in zip(self.res_by_type[key][tp+'_gt'],self.res_by_type[key][tp+'_pred']):
                    per_ans_stat[str(g)+'_gt']+=1
                    if g==p:
                        per_ans_stat[str(g)]+=1
                unq_acc = 0
                for unq_ans in set(self.res_by_type[key][tp+'_gt']):
                    acc_curr_ans = per_ans_stat[str(unq_ans)]/per_ans_stat[str(unq_ans)+'_gt']
                    unq_acc +=acc_curr_ans
                acc = 100*unq_acc/len(set(self.res_by_type[key][tp+'_gt']))
                sum_acc.append(acc+eps)
                Logger()("Accuracy {} for class '{}' is {:.2f}".format(key, tp, acc))
                Logger().log_value('{}_epoch.tdiuc{}.perQuestionType_norm.{}'.format(self.mode, key, tp), acc, should_print=False)

            acc_mpt_a = float(np.mean(np.array(sum_acc)))
            Logger()('Arithmetic MPT Accuracy is {:.2f}'.format(acc_mpt_a))
            Logger().log_value('{}_epoch.tdiuc{}.acc_mpt_a_norm'.format(self.mode, key), acc_mpt_a, should_print=False)

            acc_mpt_h = float(stats.hmean(sum_acc))
            Logger()('Harmonic MPT Accuracy is {:.2f}'.format(acc_mpt_h))
            Logger().log_value('{}_epoch.tdiuc{}.acc_mpt_h_norm'.format(self.mode, key), acc_mpt_h, should_print=False)
