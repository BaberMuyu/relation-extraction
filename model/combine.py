import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel

from global_config import *
from .hbt import DGCNN_HBT
from utils import sl_loss


class Combine(nn.Module):
    def __init__(self, schema_num, sbj_type_num):
        super(Combine, self).__init__()
        models = []
        model = DGCNN_HBT(schema_num=schema_num, sbj_type_num=sbj_type_num)
        model.load_state_dict(torch.load(ROOT_SAVED_MODEL + 'large/swa_8017.ckpt'))
        for p in model.parameters():
            p.requires_grad = False
        models.append(model)
        model = DGCNN_HBT(schema_num=schema_num, sbj_type_num=sbj_type_num)
        model.load_state_dict(torch.load(ROOT_SAVED_MODEL + 'large/swa_8012.ckpt'))
        for p in model.parameters():
            p.requires_grad = False
        models.append(model)
        self.models = nn.ModuleList(models)
        self.sbj_linear = nn.Sequential(nn.Linear(sbj_type_num * len(models) * 2, sbj_type_num * len(models) * 2),
                                        nn.Linear(sbj_type_num * len(models) * 2, sbj_type_num * 2))
        self.obj_linear = nn.Sequential(nn.Linear(schema_num * len(models) * 2, schema_num * len(models) * 2),
                                        nn.Linear(schema_num * len(models) * 2, schema_num * 2))
        self.word_linear = nn.Sequential(nn.Linear(len(models), len(models)),
                                         nn.Linear(len(models), 1))
        self.point_threshold = 0.5
        self.sbj_type_num = sbj_type_num
        self.schema_num = schema_num

    def forward(self, inputs):
        texts, texts_mask, text_select_indices, sbj_masks, w2vs, hands = inputs
        text_vecs = []
        word_vecs = []
        sbj_points_list = []
        sbj_inputs = [texts, texts_mask, w2vs, hands]
        for model in self.models:
            text_vec, word_pred, sbj_points = model.predicate_1(sbj_inputs)
            text_vec = text_vec.detach()
            word_pred = word_pred.detach()
            sbj_points = sbj_points.detach()
            text_vecs.append(text_vec)
            word_vecs.append(word_pred)
            sbj_points_list.append(sbj_points)

        sbj_points = torch.cat(sbj_points_list, dim=1).permute(0,2,1)
        word_pred = torch.stack(word_vecs, dim=2)
        sbj_points = nn.functional.sigmoid(self.sbj_linear(sbj_points).permute(0,2,1))
        word_pred = nn.functional.sigmoid(self.word_linear(word_pred).squeeze(dim=2))

        obj_points_list = []
        for text_vec, model in zip(text_vecs, self.models):
            obj_points = model.predicate_2(text_vec, texts_mask, text_select_indices, sbj_masks)
            obj_points = obj_points.detach()
            obj_points_list.append(obj_points)
        obj_points = torch.cat(obj_points_list, dim=1).permute(0,2,1)
        obj_points = nn.functional.sigmoid(self.obj_linear(obj_points).permute(0,2,1))
        return [sbj_points, obj_points, word_pred]

    def calculate_loss(self, preds, y_trues, mask, texts_select_indices):
        sbj_pred = preds[0]
        obj_pred = preds[1]
        word_pred = preds[2]

        sbj_true = y_trues[0]
        obj_true = y_trues[1]
        word_true = y_trues[2]

        A = -8
        a = 0.5
        b = 1
        c = 2
        # sbj loss
        sbj_mask = mask.unsqueeze(dim=1)
        sbj_mask_1 = sbj_mask * sbj_true
        sbj_mask_0 = sbj_mask * (1 - sbj_mask_1)
        sbj_loss = sl_loss(sbj_pred, sbj_true, A, a, b)
        # sbj_loss = sbj_loss * sbj_mask
        sbj_loss_0 = sbj_loss * sbj_mask_0
        sbj_loss_1 = sbj_loss * sbj_mask_1
        sum_0 = sbj_mask_0.sum()
        sum_1 = sbj_mask_1.sum()
        sbj_loss_0 = sbj_loss_0.sum().div(sum_0)
        sbj_loss_1 = sbj_loss_1.sum().div(sum_1)
        rate_0 = (sum_0 + sum_1 - sum_1 * c) / (sum_0 + sum_1)
        sbj_loss = sbj_loss_0 * rate_0 + sbj_loss_1 * (1 - rate_0)

        # obj loss
        obj_mask = torch.index_select(mask, 0, texts_select_indices).unsqueeze(dim=1)
        obj_loss = sl_loss(obj_pred, obj_true, A, a, b)
        obj_mask_1 = obj_mask * obj_true
        obj_mask_0 = obj_mask * (1 - obj_mask_1)
        obj_loss_0 = obj_loss * obj_mask_0
        obj_loss_1 = obj_loss * obj_mask_1
        sum_0 = obj_mask_0.sum()
        sum_1 = obj_mask_1.sum()
        obj_loss_0 = obj_loss_0.sum().div(sum_0)
        obj_loss_1 = obj_loss_1.sum().div(sum_1)
        rate_0 = (sum_0 + sum_1 - sum_1 * c) / (sum_0 + sum_1)
        obj_loss = obj_loss_0 * rate_0 + obj_loss_1 * (1 - rate_0)

        # word loss
        word_loss = sl_loss(word_pred, word_true, A, a, b)
        word_loss = word_loss * mask
        word_loss = word_loss.sum().div(mask.sum())

        loss = sbj_loss + obj_loss * 2.5 + word_loss * 0.015
        return loss

    def evaluate(self, inputs, raw_text):
        texts, texts_mask, text_select_indices, sbj_masks, w2vs, hands = inputs
        text_vecs = []
        word_vecs = []
        sbj_points_list = []

        sbj_inputs = [texts, texts_mask, w2vs, hands]
        for model in self.models:
            text_vec, word_pred, sbj_points = model.predicate_1(sbj_inputs)
            text_vecs.append(text_vec)
            word_vecs.append(word_pred)
            sbj_points_list.append(sbj_points)

        sbj_points = torch.cat(sbj_points_list, dim=1).permute(0,2,1)
        word_pred = torch.stack(word_vecs, dim=2)
        sbj_points = nn.functional.sigmoid(self.sbj_linear(sbj_points).permute(0,2,1))
        word_pred = nn.functional.sigmoid(self.word_linear(word_pred).squeeze(dim=2))

        obj_points_list = []
        for text_vec, model in zip(text_vecs, self.models):
            obj_points = model.predicate_2(text_vec, texts_mask, text_select_indices, sbj_masks)
            obj_points_list.append(obj_points)
        obj_points = torch.cat(obj_points_list, dim=1).permute(0,2,1)
        obj_points = nn.functional.sigmoid(self.obj_linear(obj_points).permute(0,2,1))

        model = self.models[0]
        sbj_masks, text_select_indices, sbj_entities, sbj_entities_point = model.predicate_1_1(text_vecs[0], sbj_points,
                                                                                               word_pred, raw_text)
        eval_obj_points_list = []
        for text_vec, model in zip(text_vecs, self.models):
            eval_obj_points = model.predicate_2(text_vec, texts_mask, text_select_indices, sbj_masks)
            eval_obj_points_list.append(eval_obj_points)
        eval_obj_points = torch.cat(eval_obj_points_list, dim=1).permute(0,2,1)
        eval_obj_points = nn.functional.sigmoid(self.obj_linear(eval_obj_points).permute(0,2,1))
        spo_list = model.predicate_2_2(eval_obj_points, text_select_indices, word_pred, sbj_entities,
                                       sbj_entities_point, raw_text)

        return [sbj_points, obj_points, word_pred], spo_list

    def predicate(self, inputs, raw_text):
        texts, texts_mask, w2vs, hands = inputs
        text_vecs = []
        word_vecs = []
        sbj_points_list = []

        sbj_inputs = [texts, texts_mask, w2vs, hands]
        for model in self.models:
            text_vec, word_pred, sbj_points = model.predicate_1(sbj_inputs)
            text_vecs.append(text_vec)
            word_vecs.append(word_pred)
            sbj_points_list.append(sbj_points)

        sbj_points = torch.cat(sbj_points_list, dim=1).permute(0,2,1)
        word_pred = torch.stack(word_vecs, dim=2)
        sbj_points = nn.functional.sigmoid(self.sbj_linear(sbj_points).permute(0,2,1))
        word_pred = nn.functional.sigmoid(self.word_linear(word_pred).squeeze(dim=2))

        model = self.models[0]
        sbj_masks, text_select_indices, sbj_entities, sbj_entities_point = model.predicate_1_1(text_vecs[0], sbj_points,
                                                                                               word_pred, raw_text)
        obj_points_list = []
        for text_vec, model in zip(text_vecs, self.models):
            obj_points = model.predicate_2(text_vec, texts_mask, text_select_indices, sbj_masks)
            obj_points_list.append(obj_points)
        obj_points = torch.cat(obj_points_list, dim=1).permute(0,2,1)
        obj_points = nn.functional.sigmoid(self.obj_linear(obj_points).permute(0,2,1))
        spo_point_list = model.predicate_2_2(obj_points, text_select_indices, word_pred, sbj_entities,
                                             sbj_entities_point, raw_text)
        return spo_point_list

    def find_entities(self, text_line, ps, pe, ps_limit_map, pe_limit_map):
        def is_cross_point(a_point, entities_points):
            start_in_flag = False
            end_in_flag = False
            for e_index in range(len(entities_points)):
                for p_index in range(len(entities_points[e_index])):
                    if not start_in_flag and entities_points[e_index][p_index][0] < a_point[0] <= \
                            entities_points[e_index][p_index][1]:
                        start_in_flag = True
                    elif not end_in_flag and entities_points[e_index][p_index][0] <= a_point[1] < \
                            entities_points[e_index][p_index][1]:
                        end_in_flag = True
                    if start_in_flag and end_in_flag:
                        return True
            return False

        entities_line = []
        entities_point_line = []
        seq_length = len(text_line)
        start_index = -999
        end_index = 999
        ps_map = np.zeros(seq_length, dtype=np.int)
        pe_map = np.zeros(seq_length, dtype=np.int)
        start_list = []
        end_list = []
        for index in range(seq_length):
            if ps[index]:
                start_index = index
                start_list.append(start_index)
            if pe[seq_length - index - 1]:
                end_index = seq_length - index - 1
                end_list.append(end_index)
            ps_map[index] = start_index
            pe_map[seq_length - index - 1] = end_index

        for start_index in start_list:
            end_index = pe_map[start_index]
            if end_index != 999:
                if end_index - start_index > 10:
                    if pe_map[start_index] > pe_limit_map[start_index] and pe_limit_map[
                        start_index] - start_index > 2:
                        end_index = pe_limit_map[start_index]
                pass
            else:
                # self.start_num += 1
                # print('only start', self.start_num)
                continue
            entity = text_line[start_index:end_index + 1]
            entity_point = (start_index, end_index)
            try:
                entity_index = entities_line.index(entity)
                if entity_point not in entities_point_line[entity_index]:
                    entities_point_line[entity_index].append(entity_point)
            except ValueError:
                entities_line.append(entity)
                entities_point_line.append([entity_point])

        for end_index in end_list:
            start_index = ps_map[end_index]
            if start_index != -999:
                if end_index - start_index > 10:
                    if ps_map[end_index] < ps_limit_map[end_index] and end_index - ps_limit_map[end_index] > 2:
                        start_index = ps_limit_map[end_index]
                pass
            else:
                # self.end_num += 1
                # print('only end', self.end_num)
                continue
            entity = text_line[start_index:end_index + 1]
            entity_point = (start_index, end_index)
            try:
                entity_index = entities_line.index(entity)
                if entity_point not in entities_point_line[entity_index]:
                    entities_point_line[entity_index].append(entity_point)
            except ValueError:
                entities_line.append(entity)
                entities_point_line.append([entity_point])

        new_entities_line = []
        new_entities_point_line = []
        for entity_index in range(len(entities_point_line)):
            for point in entities_point_line[entity_index][:]:
                if is_cross_point(point, entities_point_line):
                    del (entities_point_line[entity_index][entities_point_line[entity_index].index(point)])
            if len(entities_point_line[entity_index]) > 0:
                new_entities_line.append(entities_line[entity_index])
                new_entities_point_line.append(entities_point_line[entity_index])
            # else:
            # print(entities_line[entity_index])
            # print('cross')

        return new_entities_line, new_entities_point_line
        # return entities_line, entities_point_line

    def find_sbj_entities(self, raw_text, points, words):
        points = points.cpu().numpy()
        words = words.cpu().numpy()
        points = points > self.point_threshold
        words = words > self.point_threshold
        entities = []
        entities_point = []

        for batch_index in range(points.shape[0]):
            entities_line = []
            entities_point_line = []
            word_line = words[batch_index]
            start_index = -999
            end_index = 999
            seq_len = len(word_line)
            ps_limit_map = np.zeros(seq_len, dtype=np.int)
            pe_limit_map = np.zeros(seq_len, dtype=np.int)
            for index in range(seq_len):
                if word_line[index]:
                    if start_index == -999:
                        start_index = index
                else:
                    start_index = -999
                if word_line[seq_len - index - 1]:
                    if end_index == 999:
                        end_index = seq_len - index - 1
                else:
                    end_index = 999
                pe_limit_map[seq_len - index - 1] = end_index
                ps_limit_map[index] = start_index

            for sbj_type_index in range(self.sbj_type_num):
                eee, ppp = self.find_entities(raw_text[batch_index], points[batch_index][sbj_type_index],
                                              points[batch_index][sbj_type_index + self.sbj_type_num],
                                              ps_limit_map, pe_limit_map)
                for eeee, pppp in zip(eee, ppp):
                    if eeee not in entities_line:
                        entities_line.append(eeee)
                        entities_point_line.append(pppp)

            entities.append(entities_line)
            entities_point.append(entities_point_line)
        return entities, entities_point

    def find_obj_entities(self, raw_text, points, text_indices, words):
        points = points.cpu().numpy()
        words = words.cpu().numpy()
        points = points > self.point_threshold
        words = words > self.point_threshold
        entities = [[] for _ in range(len(raw_text))]
        entities_point = [[] for _ in range(len(raw_text))]

        for point_index in range(points.shape[0]):
            batch_index = text_indices[point_index]
            entities_line = []
            entities_point_line = []
            word_line = words[batch_index]
            start_index = -999
            end_index = 999
            seq_len = len(word_line)
            ps_limit_map = np.zeros(seq_len, dtype=np.int)
            pe_limit_map = np.zeros(seq_len, dtype=np.int)
            for index in range(seq_len):
                if word_line[index]:
                    if start_index == -999:
                        start_index = index
                else:
                    start_index = -999
                if word_line[seq_len - index - 1]:
                    if end_index == 999:
                        end_index = seq_len - index - 1
                else:
                    end_index = 999
                pe_limit_map[seq_len - index - 1] = end_index
                ps_limit_map[index] = start_index
            for schema_index in range(self.schema_num):
                eee, ppp = self.find_entities(raw_text[batch_index], points[point_index][schema_index],
                                              points[point_index][schema_index + self.schema_num],
                                              ps_limit_map, pe_limit_map)
                entities_line.append(eee)
                entities_point_line.append(ppp)

            entities[batch_index].append(entities_line)
            entities_point[batch_index].append(entities_point_line)
        return entities, entities_point

    def get_spo_list(self, sbj_entities, obj_entities, sbj_entities_point=None, obj_entities_point=None):
        spo_list = []
        spo_point_list = []
        for batch_index in range(len(sbj_entities)):
            spo_list.append([])
            spo_point_list.append([])
            for sbj_index in range(len(sbj_entities[batch_index])):
                for schema_index in range(self.schema_num):
                    for obj_index in range(len(obj_entities[batch_index][sbj_index][schema_index])):
                        spo_list[batch_index].append([sbj_entities[batch_index][sbj_index],
                                                      schema_index,
                                                      obj_entities[batch_index][sbj_index][schema_index][
                                                          obj_index]])
                        if sbj_entities_point is not None:
                            spo_point_list[batch_index].append([sbj_entities_point[batch_index][sbj_index],
                                                                schema_index,
                                                                obj_entities_point[batch_index][sbj_index][
                                                                    schema_index]
                                                                [obj_index]])
        return spo_list, spo_point_list

    def calculate_train_f1(self, raw_text, preds, y_trues, text_select_indices):
        sbj_points = preds[0]
        obj_points = preds[1]
        word_pred = preds[2]

        sbj_entities_true = y_trues[0]
        spo_true = y_trues[1]

        sbj_entities_pred, _ = self.find_sbj_entities(raw_text, sbj_points, word_pred)
        obj_entities_pred, _ = self.find_obj_entities(raw_text, obj_points, text_select_indices, word_pred)
        spo_pred, _ = self.get_spo_list(sbj_entities_true, obj_entities_pred)

        sbj_correct_num = sbj_pred_num = sbj_true_num = 0
        for batch_index in range(len(sbj_entities_pred)):
            for sbj in sbj_entities_pred[batch_index]:
                if sbj in sbj_entities_true[batch_index]:
                    sbj_correct_num += 1
            sbj_pred_num += len(sbj_entities_pred[batch_index])
            sbj_true_num += len(sbj_entities_true[batch_index])

        spo_correct_num = spo_pred_num = spo_true_num = 0
        for batch_index in range(len(spo_pred)):
            for spo in spo_pred[batch_index]:
                if spo in spo_true[batch_index]:
                    spo_correct_num += 1
            spo_pred_num += len(spo_pred[batch_index])
            spo_true_num += len(spo_true[batch_index])

        return [sbj_correct_num, sbj_pred_num, sbj_true_num], [spo_correct_num, spo_pred_num, spo_true_num]