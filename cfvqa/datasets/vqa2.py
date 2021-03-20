import os
import csv
import copy
import json
import torch
import numpy as np
from os import path as osp
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from block.datasets.vqa_utils import AbstractVQA
from copy import deepcopy
import random
import tqdm
import h5py

class VQA2(AbstractVQA):

    def __init__(self,
            dir_data='data/vqa2',
            split='train', 
            batch_size=10,
            nb_threads=4,
            pin_memory=False,
            shuffle=False,
            nans=1000,
            minwcount=10,
            nlp='mcb',
            proc_split='train',
            samplingans=False,
            dir_rcnn='data/coco/extract_rcnn',
            adversarial=False,
            dir_cnn=None
            ):

        super(VQA2, self).__init__(
            dir_data=dir_data,
            split=split,
            batch_size=batch_size,
            nb_threads=nb_threads,
            pin_memory=pin_memory,
            shuffle=shuffle,
            nans=nans,
            minwcount=minwcount,
            nlp=nlp,
            proc_split=proc_split,
            samplingans=samplingans,
            has_valset=True,
            has_testset=True,
            has_answers_occurence=True,
            do_tokenize_answers=False)            

        self.dir_rcnn = dir_rcnn
        self.dir_cnn = dir_cnn
        self.load_image_features()
        # to activate manually in visualization context (notebo# to activate manually in visualization context (notebook)
        self.load_original_annotation = False

    def add_rcnn_to_item(self, item):
        path_rcnn = os.path.join(self.dir_rcnn, '{}.pth'.format(item['image_name']))
        item_rcnn = torch.load(path_rcnn)
        item['visual'] = item_rcnn['pooled_feat']
        item['coord'] = item_rcnn['rois']
        item['norm_coord'] = item_rcnn.get('norm_rois', None)
        item['nb_regions'] = item['visual'].size(0)
        return item

    def add_cnn_to_item(self, item):
        image_name = item['image_name']
        if image_name in self.image_names_to_index_train:
            index = self.image_names_to_index_train[image_name]
            image = torch.tensor(self.image_features_train['att'][index])
        elif image_name in self.image_names_to_index_val:
            index = self.image_names_to_index_val[image_name]
            image = torch.tensor(self.image_features_val['att'][index])
        image = image.permute(1, 2, 0).view(196, 2048)
        item['visual'] = image
        return item

    def load_image_features(self):
        if self.dir_cnn:
            filename_train = os.path.join(self.dir_cnn, 'trainset.hdf5')
            filename_val = os.path.join(self.dir_cnn, 'valset.hdf5')
            Logger()(f"Opening file {filename_train}, {filename_val}")
            self.image_features_train = h5py.File(filename_train, 'r', swmr=True)
            self.image_features_val = h5py.File(filename_val, 'r', swmr=True)
            # load txt
            with open(os.path.join(self.dir_cnn, 'trainset.txt'.format(self.split)), 'r') as f:
                self.image_names_to_index_train = {}
                for i, line in enumerate(f):
                    self.image_names_to_index_train[line.strip()] = i
            with open(os.path.join(self.dir_cnn, 'valset.txt'.format(self.split)), 'r') as f:
                self.image_names_to_index_val = {}
                for i, line in enumerate(f):
                    self.image_names_to_index_val[line.strip()] = i

    def __getitem__(self, index):
        item = {}
        item['index'] = index

        # Process Question (word token)
        question = self.dataset['questions'][index]
        if self.load_original_annotation:
            item['original_question'] = question

        item['question_id'] = question['question_id']

        item['question'] = torch.tensor(question['question_wids'], dtype=torch.long)
        item['lengths'] = torch.tensor([len(question['question_wids'])], dtype=torch.long)
        item['image_name'] = question['image_name']

        # Process Object, Attribut and Relational features
        # Process Object, Attribut and Relational features
        if self.dir_rcnn:
            item = self.add_rcnn_to_item(item)
        elif self.dir_cnn:
            item = self.add_cnn_to_item(item)

        # Process Answer if exists
        if 'annotations' in self.dataset:
            annotation = self.dataset['annotations'][index]
            if self.load_original_annotation:
                item['original_annotation'] = annotation
            if 'train' in self.split and self.samplingans:
                proba = annotation['answers_count']
                proba = proba / np.sum(proba)
                item['answer_id'] = int(np.random.choice(annotation['answers_id'], p=proba))
            else:
                item['answer_id'] = annotation['answer_id']
            item['class_id'] = torch.tensor([item['answer_id']], dtype=torch.long)
            item['answer'] = annotation['answer']
            item['question_type'] = annotation['question_type']
        else:
            if item['question_id'] in self.is_qid_testdev:
                item['is_testdev'] = True
            else:
                item['is_testdev'] = False

        # if Options()['model.network.name'] == 'xmn_net':
        #     num_feat = 36
        #     relation_mask = np.zeros((num_feat, num_feat))
        #     boxes = item['coord']
        #     for i in range(num_feat):
        #         for j in range(i+1, num_feat):
        #             # if there is no overlap between two bounding box
        #             if boxes[0,i]>boxes[2,j] or boxes[0,j]>boxes[2,i] or boxes[1,i]>boxes[3,j] or boxes[1,j]>boxes[3,i]:
        #                 pass
        #             else:
        #                 relation_mask[i,j] = relation_mask[j,i] = 1
        #     relation_mask = torch.from_numpy(relation_mask).byte()
        #     item['relation_mask'] = relation_mask

        return item

    def download(self):
        dir_zip = osp.join(self.dir_raw, 'zip')
        os.system('mkdir -p '+dir_zip)
        dir_ann = osp.join(self.dir_raw, 'annotations')
        os.system('mkdir -p '+dir_ann)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P '+dir_zip)
        os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P '+dir_zip)
        os.system('unzip '+osp.join(dir_zip, 'v2_Questions_Train_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+osp.join(dir_zip, 'v2_Questions_Val_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+osp.join(dir_zip, 'v2_Questions_Test_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+osp.join(dir_zip, 'v2_Annotations_Train_mscoco.zip')+' -d '+dir_ann)
        os.system('unzip '+osp.join(dir_zip, 'v2_Annotations_Val_mscoco.zip')+' -d '+dir_ann)
        os.system('mv '+osp.join(dir_ann, 'v2_mscoco_train2014_annotations.json')+' '
                       +osp.join(dir_ann, 'mscoco_train2014_annotations.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_mscoco_val2014_annotations.json')+' '
                       +osp.join(dir_ann, 'mscoco_val2014_annotations.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_train2014_questions.json')+' '
                       +osp.join(dir_ann, 'OpenEnded_mscoco_train2014_questions.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_val2014_questions.json')+' '
                       +osp.join(dir_ann, 'OpenEnded_mscoco_val2014_questions.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_test2015_questions.json')+' '
                       +osp.join(dir_ann, 'OpenEnded_mscoco_test2015_questions.json'))
        os.system('mv '+osp.join(dir_ann, 'v2_OpenEnded_mscoco_test-dev2015_questions.json')+' '
                       +osp.join(dir_ann, 'OpenEnded_mscoco_test-dev2015_questions.json'))
