import os
import csv
import copy
import json
import torch
import numpy as np
from tqdm import tqdm
from os import path as osp
from bootstrap.lib.logger import Logger
from block.datasets.vqa_utils import AbstractVQA
from copy import deepcopy
import random
import h5py

class VQACP2(AbstractVQA):

    def __init__(self,
            dir_data='data/vqa/vqacp2',
            split='train',
            batch_size=80,
            nb_threads=4,
            pin_memory=False,
            shuffle=False,
            nans=1000,
            minwcount=10,
            nlp='mcb',
            proc_split='train',
            samplingans=False,
            dir_rcnn='data/coco/extract_rcnn',
            dir_cnn=None,
            dir_vgg16=None,
            has_testdevset=False,
            ):
        super(VQACP2, self).__init__(
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
            has_testset=False,
            has_testdevset=has_testdevset,
            has_testset_anno=False,
            has_answers_occurence=True,
            do_tokenize_answers=False)
        self.dir_rcnn = dir_rcnn
        self.dir_cnn = dir_cnn
        self.dir_vgg16 = dir_vgg16
        self.load_image_features()
        self.load_original_annotation = False

    def add_rcnn_to_item(self, item):
        path_rcnn = os.path.join(self.dir_rcnn, '{}.pth'.format(item['image_name']))
        item_rcnn = torch.load(path_rcnn)
        item['visual'] = item_rcnn['pooled_feat']
        item['coord'] = item_rcnn['rois']
        item['norm_coord'] = item_rcnn['norm_rois']
        item['nb_regions'] = item['visual'].size(0)
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
        elif self.dir_vgg16:
            # list filenames
            self.filenames_train = os.listdir(os.path.join(self.dir_vgg16, 'train'))
            self.filenames_val = os.listdir(os.path.join(self.dir_vgg16, 'val'))


    def add_vgg_to_item(self, item):
        image_name = item['image_name']
        filename = image_name + '.pth'
        if filename in self.filenames_train:
            path = os.path.join(self.dir_vgg16, 'train', filename)
        elif filename in self.filenames_val:
            path = os.path.join(self.dir_vgg16, 'val', filename)
        visual = torch.load(path)
        visual = visual.permute(1, 2, 0).view(14*14, 512)
        item['visual'] = visual
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

    def __getitem__(self, index):
        item = {}
        item['index'] = index

        # Process Question (word token)
        question = self.dataset['questions'][index]
        if self.load_original_annotation:
            item['original_question'] = question
        item['question_id'] = question['question_id']
        item['question'] = torch.LongTensor(question['question_wids'])
        item['lengths'] = torch.LongTensor([len(question['question_wids'])])
        item['image_name'] = question['image_name']

        # Process Object, Attribut and Relational features
        if self.dir_rcnn:
            item = self.add_rcnn_to_item(item)
        elif self.dir_cnn:
            item = self.add_cnn_to_item(item)
        elif self.dir_vgg16:
            item = self.add_vgg_to_item(item)

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
            item['class_id'] = torch.LongTensor([item['answer_id']])
            item['answer'] = annotation['answer']
            item['question_type'] = annotation['question_type']

        return item

    def download(self):
        dir_ann = osp.join(self.dir_raw, 'annotations')
        os.system('mkdir -p '+dir_ann)
        os.system('wget https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_questions.json -P' + dir_ann)
        os.system('wget https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_questions.json -P' + dir_ann)
        os.system('wget https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_annotations.json -P' + dir_ann)
        os.system('wget https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_annotations.json -P' + dir_ann)
        train_q = {"questions":json.load(open(osp.join(dir_ann, "vqacp_v2_train_questions.json")))}
        val_q = {"questions":json.load(open(osp.join(dir_ann, "vqacp_v2_test_questions.json")))}
        train_ann = {"annotations":json.load(open(osp.join(dir_ann, "vqacp_v2_train_annotations.json")))}
        val_ann = {"annotations":json.load(open(osp.join(dir_ann, "vqacp_v2_test_annotations.json")))}
        train_q['info'] = {}
        train_q['data_type'] = 'mscoco'
        train_q['data_subtype'] = "train2014cp"
        train_q['task_type'] = "Open-Ended"
        train_q['license'] = {}
        val_q['info'] = {}
        val_q['data_type'] = 'mscoco'
        val_q['data_subtype'] = "val2014cp"
        val_q['task_type'] = "Open-Ended"
        val_q['license'] = {}
        for k in ["info", 'data_type','data_subtype', 'license']:
            train_ann[k] = train_q[k]
            val_ann[k] = val_q[k]
        with open(osp.join(dir_ann, "OpenEnded_mscoco_train2014_questions.json"), 'w') as F:
            F.write(json.dumps(train_q))
        with open(osp.join(dir_ann, "OpenEnded_mscoco_val2014_questions.json"), 'w') as F:
            F.write(json.dumps(val_q))
        with open(osp.join(dir_ann, "mscoco_train2014_annotations.json"), 'w') as F:
            F.write(json.dumps(train_ann))
        with open(osp.join(dir_ann, "mscoco_val2014_annotations.json"), 'w') as F:
            F.write(json.dumps(val_ann))

    def add_image_names(self, dataset):
        for q in dataset['questions']:
            q['image_name'] = 'COCO_%s_%012d.jpg'%(q['coco_split'],q['image_id'])
        return dataset

