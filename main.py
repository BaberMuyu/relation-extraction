import torch
from torchcontrib.optim import SWA
import visdom
import numpy as np
import json, pickle
from collections import namedtuple
import random

from torch.utils.data import DataLoader

from utils import EarlyStopping, LearningSchedual, LogPrinter, MovingData, MyBertTokenizer
from utils import kmp
from utils.metrics import calculate_f1

from model import HBT, BaiDuBaseline, DGCNN_HBT, RNN_HBT, Combine
from data import LIC2020Env, LIC2019Env

from global_config import *

DATA_ENV = LIC2020Env


class REModelAux(object):
    def __init__(self, config, train_steps):
        self.config = config
        self.train_steps = train_steps

        self.log_printer = LogPrinter(self.config.epochs, self.train_steps)
        self.vis = visdom.Visdom(env='RE', port=40406) if False else None
        self.new_line_flag = False

    def new_line(self):
        self.new_line_flag = True

    def show_log(self, epoch, step, logs):
        global_step = epoch * self.train_steps + step
        if self.vis is not None:
            self.vis.line(X=np.reshape(np.array(global_step), [1]),
                          Y=np.reshape(logs['loss'], [1]),
                          win='loss', name='train_loss',
                          update='append' if global_step else None)
            self.vis.line(X=np.reshape(np.array(global_step), [1]),
                          Y=np.reshape(logs['dev_loss'], [1]),
                          win='loss', name='dev_loss',
                          update='append' if global_step else None)
            self.vis.line(X=np.reshape(np.array(global_step), [1]),
                          Y=np.reshape(logs['f1'], [1]),
                          win='f1', name='train_f1',
                          update='append' if global_step else None)
            self.vis.line(X=np.reshape(np.array(global_step), [1]),
                          Y=np.reshape(logs['dev_f1'], [1]),
                          win='f1', name='dev_f1',
                          update='append' if global_step else None)

        # print log and clean logs
        self.log_printer(epoch, step, logs, self.new_line_flag)
        self.new_line_flag = False


def locate_entity(text, ojbk):
    points = []
    j = 0
    while True:
        i = kmp(text, ojbk)
        if i != -1:
            text = text[i + len(ojbk):]
            i += j
            j = i + len(ojbk)
            points.append([i, j - 1])
        else:
            break
    return points


class REModelFittingBase(object):
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config

    @classmethod
    def get_collate_fn(self, mode='train'):
        def collate_fn_train(batch):
            return [], [], []

        def collate_fn_test(batch):
            return [], 0

        if mode == 'train' or mode == 'dev':
            return collate_fn_train
        elif mode == 'test':
            return collate_fn_test

    def calculate_loss(self, logits, y_trues, mask, texts_select_indices):
        return self.model.calculate_loss(logits, y_trues, mask, texts_select_indices)

    def calculate_train_f1(self, raw_text, preds, y_trues, text_select_indices):
        sbj_f1, spo_f1 = self.model.calculate_train_f1(raw_text, preds, y_trues, text_select_indices)
        return sbj_f1, spo_f1

    @classmethod
    def calculate_dev_f1(self, spo_pred, spo_true):
        spo_correct_num = spo_pred_num = spo_true_num = 0
        for batch_index in range(len(spo_pred)):
            for spo in spo_pred[batch_index]:
                if spo in spo_true[batch_index]:
                    spo_correct_num += 1
            spo_pred_num += len(spo_pred[batch_index])
            spo_true_num += len(spo_true[batch_index])
        return spo_correct_num, spo_pred_num, spo_true_num

    def train(self):
        # prepare data
        train_data = self.data('train')
        train_steps = int((len(train_data) + self.config.batch_size - 1) / self.config.batch_size)
        train_dataloader = DataLoader(train_data,
                                      batch_size=self.config.batch_size,
                                      collate_fn=self.get_collate_fn('train'),
                                      shuffle=True,
                                      num_workers=2)

        # prepare optimizer
        params_lr = [{"params": self.model.bert_parameters, 'lr': self.config.small_lr},
                     {"params": self.model.other_parameters, 'lr': self.config.large_lr}]
        optimizer = torch.optim.Adam(params_lr)
        optimizer = SWA(optimizer)

        # prepare early stopping
        early_stopping = EarlyStopping(self.model, self.config.best_model_path, big_server=BIG_GPU, mode='max',
                                       patience=10, verbose=True)

        # prepare learning schedual
        learning_schedual = LearningSchedual(optimizer, self.config.epochs, train_steps,
                                             [self.config.small_lr, self.config.large_lr])

        # prepare other
        aux = REModelAux(self.config, train_steps)
        moving_log = MovingData(window=500)

        ending_flag = False
        # self.model.load_state_dict(torch.load(ROOT_SAVED_MODEL + 'temp_model.ckpt'))
        #
        # with torch.no_grad():
        #     self.model.eval()
        #     print(self.eval())
        #     return
        for epoch in range(0, self.config.epochs):
            for step, (inputs, y_trues, spo_info) in enumerate(train_dataloader):
                inputs = [aaa.cuda() for aaa in inputs]
                y_trues = [aaa.cuda() for aaa in y_trues]
                if epoch > 0 or step == 1000:
                    self.model.detach_bert = False
                # train ================================================================================================
                preds = self.model(inputs)
                loss = self.calculate_loss(preds, y_trues, inputs[1], inputs[2])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
                optimizer.step()

                with torch.no_grad():

                    logs = {'lr0': 0, 'lr1': 0}
                    if (epoch > 0 or step > 620) and False:
                        sbj_f1, spo_f1 = self.calculate_train_f1(spo_info[0], preds, spo_info[1:3],
                                                                 inputs[2].cpu().numpy())
                        metrics_data = {'loss': loss.cpu().numpy(), 'sampled_num': 1,
                                        'sbj_correct_num': sbj_f1[0], 'sbj_pred_num': sbj_f1[1],
                                        'sbj_true_num': sbj_f1[2],
                                        'spo_correct_num': spo_f1[0], 'spo_pred_num': spo_f1[1],
                                        'spo_true_num': spo_f1[2]}
                        moving_data = moving_log(epoch * train_steps + step, metrics_data)
                        logs['loss'] = moving_data['loss'] / moving_data['sampled_num']
                        logs['sbj_precise'], logs['sbj_recall'], logs['sbj_f1'] = calculate_f1(
                            moving_data['sbj_correct_num'],
                            moving_data['sbj_pred_num'],
                            moving_data['sbj_true_num'],
                            verbose=True)
                        logs['spo_precise'], logs['spo_recall'], logs['spo_f1'] = calculate_f1(
                            moving_data['spo_correct_num'],
                            moving_data['spo_pred_num'],
                            moving_data['spo_true_num'], verbose=True)
                    else:
                        metrics_data = {'loss': loss.cpu().numpy(), 'sampled_num': 1}
                        moving_data = moving_log(epoch * train_steps + step, metrics_data)
                        logs['loss'] = moving_data['loss'] / moving_data['sampled_num']

                    # update lr
                    logs['lr0'], logs['lr1'] = learning_schedual.update_lr(epoch, step)

                    if step == int(train_steps / 2) or step + 1 == train_steps:
                        self.model.eval()
                        torch.save(self.model.state_dict(), ROOT_SAVED_MODEL + 'temp_model.ckpt')
                        aux.new_line()
                        # dev ==========================================================================================
                        dev_result = self.eval()
                        logs['dev_loss'] = dev_result['loss']
                        logs['dev_sbj_precise'] = dev_result['sbj_precise']
                        logs['dev_sbj_recall'] = dev_result['sbj_recall']
                        logs['dev_sbj_f1'] = dev_result['sbj_f1']
                        logs['dev_spo_precise'] = dev_result['spo_precise']
                        logs['dev_spo_recall'] = dev_result['spo_recall']
                        logs['dev_spo_f1'] = dev_result['spo_f1']
                        logs['dev_precise'] = dev_result['precise']
                        logs['dev_recall'] = dev_result['recall']
                        logs['dev_f1'] = dev_result['f1']

                        # other thing
                        early_stopping(logs['dev_f1'])
                        if logs['dev_f1'] > 0.730:
                            optimizer.update_swa()

                        # test =========================================================================================
                        if (epoch + 1 == self.config.epochs and step + 1 == train_steps) or early_stopping.early_stop:
                            ending_flag = True
                            optimizer.swap_swa_sgd()
                            optimizer.bn_update(train_dataloader, self.model)
                            torch.save(self.model.state_dict(), ROOT_SAVED_MODEL + 'swa.ckpt')
                            self.test(ROOT_SAVED_MODEL + 'swa.ckpt')

                        self.model.train()
                aux.show_log(epoch, step, logs)
                if ending_flag:
                    return

    def eval(self):
        dev_dataloader = DataLoader(self.data('dev'),
                                    batch_size=self.config.batch_size,
                                    collate_fn=self.get_collate_fn('dev'),
                                    num_workers=2)

        def _eval():
            nonlocal dev_dataloader
            result = {}
            metrics_data = {"loss": 0, "correct_num": 0, "pred_num": 0, "true_num": 0, "sampled_num": 0,
                            'sbj_correct_num': 0, 'sbj_pred_num': 0, 'sbj_true_num': 0,
                            'spo_correct_num': 0, 'spo_pred_num': 0, 'spo_true_num': 0}
            for inputs, y_trues, spo_info in dev_dataloader:
                inputs = [aaa.cuda() for aaa in inputs]
                y_trues = [aaa.cuda() for aaa in y_trues]
                preds, pred_spo_list = self.model.evaluate(inputs, spo_info[0])
                loss = self.calculate_loss(preds, y_trues, inputs[1], inputs[2])
                sbj_f1, spo_f1 = self.calculate_train_f1(spo_info[0], preds, spo_info[1:3], inputs[2])
                eval_spo_f1 = self.calculate_dev_f1(pred_spo_list, spo_info[2])
                metrics_data['sbj_correct_num'] += sbj_f1[0]
                metrics_data['sbj_pred_num'] += sbj_f1[1]
                metrics_data['sbj_true_num'] += sbj_f1[2]
                metrics_data['spo_correct_num'] += spo_f1[0]
                metrics_data['spo_pred_num'] += spo_f1[1]
                metrics_data['spo_true_num'] += spo_f1[2]
                metrics_data['correct_num'] += eval_spo_f1[0]
                metrics_data['pred_num'] += eval_spo_f1[1]
                metrics_data['true_num'] += eval_spo_f1[2]
                metrics_data['loss'] += loss.cpu().numpy()
                metrics_data['sampled_num'] += 1

            result['loss'] = metrics_data['loss'] / metrics_data['sampled_num']
            result['sbj_precise'], result['sbj_recall'], result['sbj_f1'] = calculate_f1(
                metrics_data['sbj_correct_num'],
                metrics_data['sbj_pred_num'],
                metrics_data['sbj_true_num'], verbose=True)
            result['spo_precise'], result['spo_recall'], result['spo_f1'] = calculate_f1(
                metrics_data['spo_correct_num'],
                metrics_data['spo_pred_num'],
                metrics_data['spo_true_num'], verbose=True)
            result['precise'], result['recall'], result['f1'] = calculate_f1(metrics_data['correct_num'],
                                                                             metrics_data['pred_num'],
                                                                             metrics_data['true_num'], verbose=True)
            return result

        return _eval()

    def test(self, model_path, mode='test1', outfile=None):
        test_dataloader = DataLoader(self.data(mode),
                                     batch_size=self.config.batch_size * 4,
                                     collate_fn=self.get_collate_fn('test'),
                                     num_workers=2)
        self.model.load_state_dict(torch.load(model_path))
        # =================================
        with torch.no_grad():
            spo_list = []
            i = 0
            for test_inputs, text in test_dataloader:
                test_inputs = [aaa.cuda() for aaa in test_inputs]
                print(i)
                i += 1
                spos = self.model.predicate(test_inputs, text)
                spo_list.extend(spos)
        outfile = mode + '_joint.json' if outfile is None else outfile
        DATA_ENV.generate_formal_results(spo_list, mode, outfile)
        return spo_list

    def test_map(self):
        test_dataloader = DataLoader(self.data('dev'),
                                     batch_size=self.config.batch_size * 2,
                                     collate_fn=self.get_collate_fn('dev'),
                                     num_workers=2)

        with torch.no_grad():
            spo_list = []
            metrics_data = {'sbj_correct_num': 0, 'sbj_pred_num': 0, 'sbj_true_num': 0,
                            'spo_correct_num': 0, 'spo_pred_num': 0, 'spo_true_num': 0}
            i = 1
            for test_inputs, y_trues, spo_info in test_dataloader:
                print(i)
                i += 1
                sbj_f1, spo_f1 = self.model.calculate_train_f1(spo_info[0], y_trues, spo_info[1:3], test_inputs[2])
                metrics_data['sbj_correct_num'] += sbj_f1[0]
                metrics_data['sbj_pred_num'] += sbj_f1[1]
                metrics_data['sbj_true_num'] += sbj_f1[2]
                metrics_data['spo_correct_num'] += spo_f1[0]
                metrics_data['spo_pred_num'] += spo_f1[1]
                metrics_data['spo_true_num'] += spo_f1[2]
            print(calculate_f1(metrics_data['sbj_correct_num'], metrics_data['sbj_pred_num'],
                               metrics_data['sbj_true_num'], verbose=True))
            print(calculate_f1(metrics_data['spo_correct_num'], metrics_data['spo_pred_num'],
                               metrics_data['spo_true_num'], verbose=True))
        return spo_list

    def test_a_line(self, model_path):
        tokenizer = MyBertTokenizer.from_pretrained(BERT_MODEL)
        schema = DATA_ENV.Schema()
        self.model.load_state_dict(torch.load(model_path))
        inputs_string = '"《正道沧桑——社会主义500年》的主题曲《梦想》阐述了“中国梦，幸福路”，是第一首唱响中国梦的歌曲"'
        with torch.no_grad():
            while True:
                text = tokenizer.encode(inputs_string, max_length=DATA_ENV.MAX_LENGTH)
                text_list = [text]
                mask = torch.ones(1, len(text))

                text = torch.tensor(text_list)
                spo_point = self.model.predicate((text.cuda(), mask.float().cuda()), text_list)
                result = DATA_ENV.get_formal_result(inputs_string, spo_point[0], tokenizer, schema, post=False)
                print(result)
                inputs_string = input("input your text")
                if inputs_string == 'quit':
                    break


class REModelFittingHBT(REModelFittingBase):
    def __init__(self, model, data, config):
        super(REModelFittingHBT, self).__init__(model, data, config)

    @classmethod
    def get_collate_fn(self, mode='train'):
        def collate_fn_train(batch):
            max_length = 0

            raw_texts = []
            sbj_list = []
            spo_list = []
            texts = []
            texts_mask = []
            texts_select_indices = []
            word_list = []
            schema_list = []
            sbj_mask_list = []
            sbj_point_list = []
            obj_point_list = []

            for pair in batch:
                max_length = max_length if max_length > len(pair['text']) else len(pair['text'])

            for batch_index, pair in enumerate(batch):
                text_length = len(pair['text'])
                texts_mask.append([1] * text_length + [0] * (max_length - text_length))
                raw_texts.append(pair['text'])
                pair_text = pair['text'] + [0] * (max_length - text_length)
                texts.append(pair_text)

                sbjs = []
                spos = []
                words = np.zeros(max_length)
                schemas = np.zeros(DATA_ENV.NUM_SCHEMA)
                sbj_points = np.zeros([DATA_ENV.NUM_SBJ_TYPE * 2, max_length])
                obj_points = []
                for spo in pair['spo_list']:
                    schemas[spo['predicate']] = 1
                    if spo['subject'][1] not in sbjs:
                        texts_select_indices.append(batch_index)
                        sbj_mask = np.zeros(max_length)

                        sbjs.append(spo['subject'][1])
                        points = locate_entity(pair_text, spo['subject'][1])
                        for s, e in points:
                            words[s:e + 1] = 1
                            sbj_points[spo['subject'][0]][s] = 1
                            sbj_points[spo['subject'][0] + DATA_ENV.NUM_SBJ_TYPE][e] = 1
                            sbj_mask[s:e + 1] = 1
                        sbj_mask_list.append(sbj_mask)

                        obj_point = np.zeros([DATA_ENV.NUM_SCHEMA * 2, max_length])
                        for obj_key in spo['object'].keys():
                            spos.append([spo['subject'][1], spo['predicate'], spo['object'][obj_key][1]])
                            points = locate_entity(pair_text, spo['object'][obj_key][1])
                            for s, e in points:
                                words[s:e + 1] = 1
                                obj_point[spo['predicate']][s] = 1
                                obj_point[spo['predicate'] + DATA_ENV.NUM_SCHEMA][e] = 1
                        obj_points.append(obj_point)
                    else:
                        sbj_index = sbjs.index(spo['subject'][1])
                        for obj_key in spo['object'].keys():
                            spos.append([spo['subject'][1], spo['predicate'], spo['object'][obj_key][1]])
                            points = locate_entity(pair_text, spo['object'][obj_key][1])
                            for s, e in points:
                                words[s:e + 1] = 1
                                obj_points[sbj_index][spo['predicate']][s] = 1
                                obj_points[sbj_index][spo['predicate'] + DATA_ENV.NUM_SCHEMA][e] = 1

                sbj_list.append(sbjs)
                spo_list.append(spos)
                word_list.append(words)
                schema_list.append(schemas)
                sbj_point_list.append(sbj_points)
                obj_point_list.extend(obj_points)

            texts = torch.tensor(texts)
            texts_mask = torch.tensor(texts_mask).float()
            texts_select_indices = torch.tensor(texts_select_indices).long()
            sbj_mask_list = torch.tensor(sbj_mask_list).float()
            sbj_point_list = torch.tensor(sbj_point_list).float()
            obj_point_list = torch.tensor(obj_point_list).float()
            word_list = torch.tensor(word_list).float()
            schema_list = torch.tensor(schema_list).float()
            #          masks 保持在最后一个
            return [texts, texts_mask, texts_select_indices, sbj_mask_list], \
                   [sbj_point_list, obj_point_list, word_list, schema_list], \
                   [raw_texts, sbj_list, spo_list]

        def collate_fn_test(batch):
            max_length = 0
            texts = []
            raw_texts = []
            masks = []
            for pair in batch:
                max_length = max_length if max_length > len(pair['text']) else len(pair['text'])
                raw_texts.append(pair['text'])

            for pair in batch:
                text_length = len(pair['text'])
                masks.append([1] * text_length + [0] * (max_length - text_length))
                pair_text = pair['text'] + [0] * (max_length - text_length)
                texts.append(pair_text)

            texts = torch.tensor(texts)
            masks = torch.tensor(masks).float()
            return [texts, masks], raw_texts

        if mode == 'train' or mode == 'dev':
            return collate_fn_train
        elif mode == 'test':
            return collate_fn_test


class REModelFittingDGCHBT(REModelFittingBase):
    def __init__(self, model, data, config, vec_list):
        super(REModelFittingDGCHBT, self).__init__(model, data, config)
        self.vec_list = vec_list

    def get_collate_fn(self, mode='train'):
        def collate_fn_train(batch):
            max_length = 0

            raw_texts = []
            sbj_list = []
            spo_list = []
            texts = []
            word2vecs = []
            hands = []
            texts_mask = []
            texts_select_indices = []
            word_list = []
            schema_list = []
            sbj_mask_list = []
            sbj_point_list = []
            obj_point_list = []

            for pair in batch:
                max_length = max_length if max_length > len(pair['text']) else len(pair['text'])

            for batch_index, pair in enumerate(batch):
                text_length = len(pair['text'])
                texts_mask.append([1] * text_length + [0] * (max_length - text_length))
                raw_texts.append(pair['text'])
                pair_text = pair['text'] + [0] * (max_length - text_length)
                word2vec = pair['b2w'] + [0] * (max_length - text_length)
                hand = pair['hand'] + [0] * (max_length - text_length)
                texts.append(pair_text)
                hands.append(hand)
                word2vecs.append([self.vec_list[w] for w in word2vec])

                sbjs = []
                spos = []
                words = np.zeros(max_length)
                schemas = np.zeros(DATA_ENV.NUM_SCHEMA)
                sbj_points = np.zeros([DATA_ENV.NUM_SBJ_TYPE * 2, max_length])
                obj_points = []
                for spo in pair['spo_list']:
                    schemas[spo['predicate']] = 1
                    if spo['subject'][1] not in sbjs:
                        texts_select_indices.append(batch_index)
                        sbj_mask = np.zeros(max_length)

                        sbjs.append(spo['subject'][1])
                        points = locate_entity(pair_text, spo['subject'][1])
                        for s, e in points:
                            words[s:e + 1] = 1
                            sbj_points[spo['subject'][0]][s] = 1
                            sbj_points[spo['subject'][0] + DATA_ENV.NUM_SBJ_TYPE][e] = 1
                            sbj_mask[s:e + 1] = 1
                        sbj_mask_list.append(sbj_mask)

                        obj_point = np.zeros([DATA_ENV.NUM_SCHEMA * 2, max_length])
                        for obj_key in spo['object'].keys():
                            spos.append([spo['subject'][1], spo['predicate'], spo['object'][obj_key][1]])
                            points = locate_entity(pair_text, spo['object'][obj_key][1])
                            for s, e in points:
                                words[s:e + 1] = 1
                                obj_point[spo['predicate']][s] = 1
                                obj_point[spo['predicate'] + DATA_ENV.NUM_SCHEMA][e] = 1
                        obj_points.append(obj_point)
                    else:
                        sbj_index = sbjs.index(spo['subject'][1])
                        for obj_key in spo['object'].keys():
                            spos.append([spo['subject'][1], spo['predicate'], spo['object'][obj_key][1]])
                            points = locate_entity(pair_text, spo['object'][obj_key][1])
                            for s, e in points:
                                words[s:e + 1] = 1
                                obj_points[sbj_index][spo['predicate']][s] = 1
                                obj_points[sbj_index][spo['predicate'] + DATA_ENV.NUM_SCHEMA][e] = 1

                sbj_list.append(sbjs)
                spo_list.append(spos)
                word_list.append(words)
                schema_list.append(schemas)
                sbj_point_list.append(sbj_points)
                obj_point_list.extend(obj_points)

            texts = torch.tensor(texts)
            word2vecs = torch.tensor(word2vecs)
            hands = torch.tensor(hands).float()
            texts_mask = torch.tensor(texts_mask).float()
            texts_select_indices = torch.tensor(texts_select_indices).long()
            sbj_mask_list = torch.tensor(sbj_mask_list).float()
            sbj_point_list = torch.tensor(sbj_point_list).float()
            obj_point_list = torch.tensor(obj_point_list).float()
            word_list = torch.tensor(word_list).float()
            schema_list = torch.tensor(schema_list).float()
            #          masks 保持在最后一个
            return [texts, texts_mask, texts_select_indices, sbj_mask_list, word2vecs, hands], \
                   [sbj_point_list, obj_point_list, word_list, schema_list], \
                   [raw_texts, sbj_list, spo_list]

        def collate_fn_test(batch):
            max_length = 0
            texts = []
            word2vecs = []
            hands = []
            raw_texts = []
            masks = []
            for pair in batch:
                max_length = max_length if max_length > len(pair['text']) else len(pair['text'])
                raw_texts.append(pair['text'])

            for pair in batch:
                text_length = len(pair['text'])
                masks.append([1] * text_length + [0] * (max_length - text_length))
                pair_text = pair['text'] + [0] * (max_length - text_length)
                word2vec = pair['b2w'] + [0] * (max_length - text_length)
                hand = pair['hand'] + [0] * (max_length - text_length)
                texts.append(pair_text)
                word2vecs.append([self.vec_list[w] for w in word2vec])
                hands.append(hand)

            texts = torch.tensor(texts)
            word2vecs = torch.tensor(word2vecs)
            hands = torch.tensor(hands).float()
            masks = torch.tensor(masks).float()
            return [texts, masks, word2vecs, hands], raw_texts

        if mode == 'train' or mode == 'dev':
            return collate_fn_train
        elif mode == 'test':
            return collate_fn_test


class CombineFittng(REModelFittingDGCHBT):
    def train(self):
        # prepare data
        train_data = self.data('dev')
        train_steps = int((len(train_data) + self.config.batch_size - 1) / self.config.batch_size)
        train_dataloader = DataLoader(train_data,
                                      batch_size=self.config.batch_size,
                                      collate_fn=self.get_collate_fn('train'),
                                      shuffle=True,
                                      num_workers=4)

        # prepare optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.large_lr)

        # prepare other
        aux = REModelAux(self.config, train_steps)
        moving_log = MovingData(window=500)

        ending_flag = False
        self.model.load_state_dict(torch.load(ROOT_SAVED_MODEL + 'temp_model.ckpt'))
        #
        # with torch.no_grad():
        #     self.model.eval()
        #     print(self.eval())
        #     return
        for epoch in range(0, self.config.epochs):
            for step, (inputs, y_trues, spo_info) in enumerate(train_dataloader):
                inputs = [aaa.cuda() for aaa in inputs]
                y_trues = [aaa.cuda() for aaa in y_trues]
                # train ================================================================================================
                preds = self.model(inputs)
                loss = self.calculate_loss(preds, y_trues, inputs[1], inputs[2])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
                optimizer.step()

                with torch.no_grad():

                    logs = {'lr0': self.config.large_lr, 'lr1': 0}
                    if epoch > 0 or step > 500:
                        sbj_f1, spo_f1 = self.calculate_train_f1(spo_info[0], preds, spo_info[1:3],
                                                                 inputs[2].cpu().numpy())
                        metrics_data = {'loss': loss.cpu().numpy(), 'sampled_num': 1,
                                        'sbj_correct_num': sbj_f1[0], 'sbj_pred_num': sbj_f1[1],
                                        'sbj_true_num': sbj_f1[2],
                                        'spo_correct_num': spo_f1[0], 'spo_pred_num': spo_f1[1],
                                        'spo_true_num': spo_f1[2]}
                        moving_data = moving_log(epoch * train_steps + step, metrics_data)
                        logs['loss'] = moving_data['loss'] / moving_data['sampled_num']
                        logs['sbj_precise'], logs['sbj_recall'], logs['sbj_f1'] = calculate_f1(
                            moving_data['sbj_correct_num'],
                            moving_data['sbj_pred_num'],
                            moving_data['sbj_true_num'],
                            verbose=True)
                        logs['spo_precise'], logs['spo_recall'], logs['spo_f1'] = calculate_f1(
                            moving_data['spo_correct_num'],
                            moving_data['spo_pred_num'],
                            moving_data['spo_true_num'], verbose=True)
                    else:
                        metrics_data = {'loss': loss.cpu().numpy(), 'sampled_num': 1}
                        moving_data = moving_log(epoch * train_steps + step, metrics_data)
                        logs['loss'] = moving_data['loss'] / moving_data['sampled_num']
                    if step + 1 == train_steps:
                        self.model.eval()
                        torch.save(self.model.state_dict(), ROOT_SAVED_MODEL + 'temp_model.ckpt')
                        aux.new_line()
                        # dev ==========================================================================================
                        # dev_result = self.eval()
                        # logs['dev_loss'] = dev_result['loss']
                        # logs['dev_sbj_precise'] = dev_result['sbj_precise']
                        # logs['dev_sbj_recall'] = dev_result['sbj_recall']
                        # logs['dev_sbj_f1'] = dev_result['sbj_f1']
                        # logs['dev_spo_precise'] = dev_result['spo_precise']
                        # logs['dev_spo_recall'] = dev_result['spo_recall']
                        # logs['dev_spo_f1'] = dev_result['spo_f1']
                        # logs['dev_precise'] = dev_result['precise']
                        # logs['dev_recall'] = dev_result['recall']
                        # logs['dev_f1'] = dev_result['f1']

                        # test =========================================================================================
                        if (epoch + 1 == self.config.epochs and step + 1 == train_steps):
                            ending_flag = True
                            self.test(ROOT_SAVED_MODEL + 'temp_model.ckpt')

                        self.model.train()
                aux.show_log(epoch, step, logs)
                if ending_flag:
                    return


# =====================================================================================================

ModelConfig = namedtuple('ModelConfig', ['epochs', 'batch_size', 'small_lr', 'large_lr', 'best_model_path'])


def predicate(vec_list, outfile, test_part='1'):
    def collate_fn_test(batch):
        max_length = 0
        texts = []
        word2vecs = []
        hands = []
        raw_texts = []
        masks = []
        for pair in batch:
            max_length = max_length if max_length > len(pair['text']) else len(pair['text'])
            raw_texts.append(pair['text'])

        for pair in batch:
            text_length = len(pair['text'])
            masks.append([1] * text_length + [0] * (max_length - text_length))
            pair_text = pair['text'] + [0] * (max_length - text_length)
            word2vec = pair['b2w'] + [0] * (max_length - text_length)
            hand = pair['hand'] + [0] * (max_length - text_length)
            texts.append(pair_text)
            word2vecs.append([vec_list[w] for w in word2vec])
            hands.append(hand)

        texts = torch.tensor(texts)
        word2vecs = torch.tensor(word2vecs)
        hands = torch.tensor(hands).float()
        masks = torch.tensor(masks).float()
        return [texts, masks, word2vecs, hands], raw_texts

    models = []
    model = DGCNN_HBT(schema_num=DATA_ENV.NUM_SCHEMA, sbj_type_num=DATA_ENV.NUM_SBJ_TYPE).cuda()
    model.load_state_dict(torch.load(ROOT_SAVED_MODEL + 'large/swa_8017.ckpt'))
    models.append(model)
    model = DGCNN_HBT(schema_num=DATA_ENV.NUM_SCHEMA, sbj_type_num=DATA_ENV.NUM_SBJ_TYPE).cuda()
    model.load_state_dict(torch.load(ROOT_SAVED_MODEL + 'large/swa_8012.ckpt'))
    models.append(model)
    model = DGCNN_HBT(schema_num=DATA_ENV.NUM_SCHEMA, sbj_type_num=DATA_ENV.NUM_SBJ_TYPE, enable_hands=True).cuda()
    model.load_state_dict(torch.load(ROOT_SAVED_MODEL + 'large/swa_7993.ckpt'))
    models.append(model)
    model = DGCNN_HBT(schema_num=DATA_ENV.NUM_SCHEMA, sbj_type_num=DATA_ENV.NUM_SBJ_TYPE, enable_hands=True).cuda()
    model.load_state_dict(torch.load(ROOT_SAVED_MODEL + 'large/swa_8040.ckpt'))
    models.append(model)
    model = DGCNN_HBT(schema_num=DATA_ENV.NUM_SCHEMA, sbj_type_num=DATA_ENV.NUM_SBJ_TYPE, enable_hands=True).cuda()
    model.load_state_dict(torch.load(ROOT_SAVED_MODEL + 'large/swa_new_2.ckpt'))
    models.append(model)

    test_dataloader = DataLoader(DATA_ENV.Data('test' + test_part),
                                 batch_size=16,
                                 collate_fn=collate_fn_test,
                                 num_workers=4)
    with torch.no_grad():
        spo_list = []
        i = 0
        for test_inputs, text in test_dataloader:
            print(i)
            i += 1
            test_inputs = [aaa.cuda() for aaa in test_inputs]
            texts, texts_mask, w2vs, hands = test_inputs
            text_vec_list = []
            sbj_ave = 0
            word_pred_ave = 0
            obj_ave = 0
            model = models[0]
            # weights = [0.13, 0.49, 0.12, 0.39]
            weights = [0.13, 0.49, 0.12, 0.13, 0.13]
            # weights = [0.01, 0.17, 0.17, 0.48, 0.17]
            for m, w in zip(models, weights):
                text_vec, word_pred, sbj_points = m.predicate_1(test_inputs)
                text_vec_list.append(text_vec)
                sbj_ave += sbj_points * w
                word_pred_ave += word_pred * w

            sbjs_mask, text_select_indices, sbj_entities, sbj_entities_point = model.predicate_1_1(text_vec, sbj_ave,
                                                                                                   word_pred_ave,
                                                                                                   text)
            for text_vec, m, w in zip(text_vec_list, models, weights):
                obj_ave += m.predicate_2(text_vec, texts_mask, text_select_indices, sbjs_mask) * w
            spo_point_list = model.predicate_2_2(obj_ave, text_select_indices, word_pred_ave, sbj_entities,
                                                 sbj_entities_point,
                                                 text)
            spo_list.extend(spo_point_list)
    DATA_ENV.generate_formal_results(spo_list, "test" + test_part, outfile)
    return spo_list


def combine_predicate(vec_list, part='1'):
    test_part = part
    result_file_list = []
    # dgc_model = REModelFittingDGCHBT(
    #     model=DGCNN_HBT(schema_num=DATA_ENV.NUM_SCHEMA, sbj_type_num=DATA_ENV.NUM_SBJ_TYPE).cuda(),
    #     data=DATA_ENV.Data,
    #     config=ModelConfig(epochs=10, batch_size=16, small_lr=0.00003, large_lr=0.00009,
    #                        best_model_path=ROOT_SAVED_MODEL + 'best.ckpt'),
    #     vec_list=vec_list)
    dgc_model_1 = REModelFittingDGCHBT(
        model=DGCNN_HBT(schema_num=DATA_ENV.NUM_SCHEMA, sbj_type_num=DATA_ENV.NUM_SBJ_TYPE, enable_hands=True).cuda(),
        data=DATA_ENV.Data,
        config=ModelConfig(epochs=10, batch_size=16, small_lr=0.00003, large_lr=0.00009,
                           best_model_path=ROOT_SAVED_MODEL + 'best.ckpt'),
        vec_list=vec_list)

    # dgc_model.test(ROOT_SAVED_MODEL + 'large/swa_8017.ckpt', mode='test' + test_part,
    #                outfile='8017_{}.json'.format(test_part))
    result_file_list.append('8017_{}.json'.format(test_part))

    # dgc_model.test(ROOT_SAVED_MODEL + 'large/swa_8012.ckpt', mode='test' + test_part,
    #                outfile='8012_{}.json'.format(test_part))
    result_file_list.append('8012_{}.json'.format(test_part))

    # dgc_model_1.test(ROOT_SAVED_MODEL + 'large/swa_7993.ckpt', mode='test' + test_part,
    #                  outfile='7993_{}.json'.format(test_part))
    result_file_list.append('7993_{}.json'.format(test_part))

    # dgc_model_1.test(ROOT_SAVED_MODEL + 'large/swa_8040.ckpt', mode='test' + test_part,
    #                  outfile='8040_{}.json'.format(test_part))
    result_file_list.append('8040_{}.json'.format(test_part))

    dgc_model_1.test(ROOT_SAVED_MODEL + 'large/swa_new_2.ckpt', mode='test' + test_part,
                     outfile='new_{}.json'.format(test_part))
    result_file_list.append('new_{}.json'.format(test_part))

    predicate(vec_list, '8040_8017_8012_7993_new{}.json'.format(test_part), test_part=test_part)

    DATA_ENV.combine_results([ROOT_RESULT + f for f in result_file_list],
                             ROOT_RESULT + '8040_8017_8012_7993_new{}.json'.format(test_part),
                             ROOT_RESULT + 'combine_result_{}.json'.format(test_part))


if __name__ == '__main__':
    vec_list = pickle.load(open(ROOT_DATA + 'w2v_vector.pkl', 'rb'))
    re_model = REModelFittingDGCHBT(
        model=DGCNN_HBT(schema_num=DATA_ENV.NUM_SCHEMA, sbj_type_num=DATA_ENV.NUM_SBJ_TYPE, enable_hands=True).cuda(),
        data=DATA_ENV.Data,
        vec_list=vec_list,
        config=ModelConfig(epochs=10,
                           batch_size=16,
                           small_lr=0.00003,
                           large_lr=0.00009,
                           best_model_path=ROOT_SAVED_MODEL + 'best.ckpt'))
    # re_model.test(ROOT_SAVED_MODEL + 'large/swa_8040.ckpt', mode='dev', outfile='dev_joint.json')
    re_model.train()
    # combine_predicate(vec_list, part='1')
