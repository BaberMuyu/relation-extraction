import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel

from global_config import BERT_MODEL, BIG_GPU
from utils import sl_loss


class HotPoint(nn.Module):
    def __init__(self):
        super(HotPoint, self).__init__()
        self.linear0 = nn.Linear(1024, 128) if BIG_GPU else nn.Linear(768, 128)
        self.linear = nn.Linear(128 * 2, 1)

    def forward(self, text, mask):
        text = self.linear0(text)
        text = nn.functional.relu(text)
        text_length = text.size()[1]
        repeat_text1 = text.unsqueeze(dim=2)
        repeat_text1 = repeat_text1.repeat(1, 1, text_length, 1)
        repeat_text2 = text.unsqueeze(dim=1)
        repeat_text2 = repeat_text2.repeat(1, text_length, 1, 1)
        repeat_text = torch.cat([repeat_text1, repeat_text2], dim=-1)
        # repeat_text = repeat_text2 - repeat_text1
        logits = self.linear(repeat_text).squeeze(dim=-1)
        logits = nn.functional.sigmoid(logits)
        return logits


class BaiDuBaseline(nn.Module):
    def __init__(self, schema_num):
        super(BaiDuBaseline, self).__init__()
        self.schema_num = schema_num
        self.bert = BertModel.from_pretrained(BERT_MODEL, output_hidden_states=True)
        # self.bert = nn.Embedding(21128, 768, 0)
        for p in self.bert.parameters():
            p.requires_grad = True

        # self.linear = nn.Linear(768, schema_num * 2 + 2)

        self.linear = nn.Linear(1024, 256) if BIG_GPU else nn.Linear(768, 256)
        self.linear_1 = nn.Linear(256, schema_num * 2 + 1)
        self.linear_word = nn.Linear(1024, 1) if BIG_GPU else nn.Linear(768, 1)

        self.linear_schema = nn.Linear(1024, schema_num) if BIG_GPU else nn.Linear(768, schema_num)
        self.hot_point = HotPoint()

        self.bert_parameters = [p for p in self.bert.parameters()]
        self.other_parameters = [p for p in self.linear.parameters()]
        self.other_parameters.extend([p for p in self.linear_1.parameters()])
        self.other_parameters.extend([p for p in self.linear_word.parameters()])
        self.other_parameters.extend([p for p in self.hot_point.parameters()])
        self.other_parameters.extend([p for p in self.linear_schema.parameters()])
        self.detach_bert = True

        self.point_threshold = 0.5

    def forward(self, inputs):
        """
        :param inputs:
                inputs[0]: text token(padding)
                inputs[1]: hot point [BATCH, LENGTH, LENGTH]
                inputs[2]: text mask
        :return:
        """
        text_token = inputs[0]
        mask = inputs[-1]
        # text_vec = self.bert(text_token)
        _, _, text_vecs = self.bert(text_token, mask)
        text_vec = text_vecs[24] if BIG_GPU else text_vecs[12]
        word_vec = text_vecs[18] if BIG_GPU else text_vecs[8]
        if self.detach_bert:
            text_vec = text_vec.detach()
            word_vec = word_vec.detach()

        logits = self.linear(text_vec)
        logits = nn.functional.relu(logits)
        logits = self.linear_1(logits)
        logits = nn.functional.sigmoid(logits)
        logits = logits.permute(0, 2, 1)

        word_pred = self.linear_word(word_vec).squeeze(-1)
        word_pred = nn.functional.sigmoid(word_pred)

        hot_points = self.hot_point(text_vec, mask)
        #
        schema_pred = self.linear_schema(text_vec[:, 0, :]).squeeze(-1)
        schema_pred = nn.functional.sigmoid(schema_pred)
        return [logits, word_pred, hot_points, schema_pred]

    def calculate_loss(self, preds, y_trues, mask):
        schema_pred = preds[3]
        hot_point = preds[2]
        word_pred = preds[1]
        logits = preds[0]

        schema_true = y_trues[3]
        hot_point_true = y_trues[2]
        word_true = y_trues[1]
        y_true = y_trues[0]

        # logits loss
        # logits_mask = mask.unsqueeze(dim=1)
        # logits_loss = nn.functional.binary_cross_entropy(torch.pow(logits, 1.3), y_true, reduction="none")
        # logits_loss = logits_loss * logits_mask
        # logits_loss = logits_loss.sum().div(logits_mask.sum() * (self.schema_num * 2 + 2))

        A = -8
        a = 0.3
        b = 1
        logits = torch.pow(logits, 3)
        logits_loss = sl_loss(logits, y_true, A, a, b)
        # logits_loss = ce_loss

        logits_mask = mask.unsqueeze(dim=1)
        logits_loss = logits_loss * logits_mask
        logits_loss = logits_loss.sum().div(logits_mask.sum() * (self.schema_num * 2 + 1))

        # word loss
        word_loss = sl_loss(word_pred, word_true, A, a, b)
        word_loss = word_loss * mask
        word_loss = word_loss.sum().div(mask.sum())

        # hot_point loss
        hot_point = torch.pow(hot_point, 4)
        hot_point_loss = sl_loss(hot_point, hot_point_true, A, a, b)
        hot_point_mask = mask.unsqueeze(dim=1) * mask.unsqueeze(dim=2)
        hot_point_loss = (hot_point_loss * hot_point_mask).sum().div(hot_point_mask.sum())

        # schema loss
        schema_loss = sl_loss(schema_pred, schema_true, A, a, b)
        schema_loss = schema_loss.mean()

        loss = logits_loss + 0.01 * word_loss + hot_point_loss * 1.5 + schema_loss * 0.01
        return loss, [logits_loss]

    def predicate(self, inputs, raw_text):
        preds = self.forward(inputs)
        spo_list, spo_point_list = self.get_spo_list(raw_text, preds)
        return spo_list, spo_point_list

    def get_spo_list(self, text, preds):
        def find_entities(text_line, head, body, mode='start'):
            entities_point = []
            entities = []
            if mode == 'start':
                for head_index in range(len(text_line)):
                    if head[head_index]:
                        start_point = head_index
                        end_point = head_index
                        for body_index in range(start_point + 1, len(text_line)):
                            if body[body_index]:
                                end_point = body_index
                            else:
                                break
                        entity = text_line[start_point:end_point + 1]
                        if entity in entities:
                            entity_index = entities.index(entity)
                            entities_point[entity_index].append([start_point, end_point])
                        else:
                            entities.append(entity)
                            entities_point.append([[start_point, end_point]])
            elif mode == 'end':
                for head_index in range(len(text_line) - 1, -1, -1):
                    if head[head_index]:
                        start_point = head_index
                        end_point = head_index
                        for body_index in range(end_point - 1, -1, -1):
                            if body[body_index]:
                                start_point = body_index
                            else:
                                break
                        entity = text_line[start_point:end_point + 1]
                        if entity in entities:
                            entity_index = entities.index(entity)
                            entities_point[entity_index].append([start_point, end_point])
                        else:
                            entities.append(entity)
                            entities_point.append([[start_point, end_point]])
            else:
                assert False
            return entities, entities_point

        def find_spo_by_hp(sbj_entities, obj_entities, sbj_entities_point, obj_entities_point, point_map, mode='start'):
            sbj_obj = []
            sbj_obj_point = []
            for sbj_index, sbj_points in enumerate(sbj_entities_point):
                for obj_index, obj_points in enumerate(obj_entities_point):
                    flag = False
                    for sbj_point_index, sbj_point in enumerate(sbj_points):
                        for obj_point_index, obj_point in enumerate(obj_points):
                            if sbj_point == obj_point:
                                continue
                            sbj_hot_point, obj_hot_point = (sbj_point[0], obj_point[0]) if mode == 'start' else (
                                sbj_point[1], obj_point[1])
                            if point_map[sbj_hot_point][obj_hot_point]:
                                sbj_obj.append([sbj_entities[sbj_index], obj_entities[obj_index]])
                                sbj_obj_point.append([sbj_entities_point[sbj_index][sbj_point_index],
                                                      obj_entities_point[obj_index][obj_point_index]])
                                flag = True
                                break
                        if flag:
                            break
            return sbj_obj, sbj_obj_point

        def find_spo_by_artificial_rules(sbj_entities, obj_entities, sbj_entities_point, obj_entities_point):
            def is_sequence(entities_0, entities_1, entities_point_0, entities_point_1):
                flag = True
                if len(entities_0) == len(entities_1):
                    entities = entities_point_0 + entities_point_1
                    for e in entities:
                        if len(e) > 1:
                            flag = False
                else:
                    flag = False
                return flag

            sbj_obj = []
            sbj_obj_point = []
            if len(sbj_entities) == 1:
                for sbj_index, sbj_points in enumerate(sbj_entities_point):
                    for obj_index, obj_points in enumerate(obj_entities_point):
                        flag = False
                        for sbj_point_index, sbj_point in enumerate(sbj_points):
                            for obj_point_index, obj_point in enumerate(obj_points):
                                if sbj_point != obj_point:
                                    sbj_obj.append([sbj_entities[sbj_index], obj_entities[obj_index]])
                                    sbj_obj_point.append([sbj_entities_point[sbj_index][sbj_point_index],
                                                          obj_entities_point[obj_index][obj_point_index]])
                                    flag = True
                                    break
                            if flag:
                                break
            elif len(obj_entities) == 1:
                for obj_index, obj_points in enumerate(obj_entities_point):
                    for sbj_index, sbj_points in enumerate(sbj_entities_point):
                        flag = False
                        for obj_point_index, obj_point in enumerate(obj_points):
                            for sbj_point_index, sbj_point in enumerate(sbj_points):
                                if sbj_point != obj_point:
                                    sbj_obj.append([sbj_entities[sbj_index], obj_entities[obj_index]])
                                    sbj_obj_point.append([sbj_entities_point[sbj_index][sbj_point_index],
                                                          obj_entities_point[obj_index][obj_point_index]])
                                    flag = True
                                    break
                            if flag:
                                break
            elif is_sequence(sbj_entities, obj_entities, sbj_entities_point, obj_entities_point):
                entity_index = 0
                while entity_index < len(sbj_entities):
                    if sbj_entities_point[entity_index][0] == obj_entities_point[entity_index][0]:
                        if entity_index != (len(sbj_entities) - 1):
                            sbj_obj.append([sbj_entities[entity_index], obj_entities[entity_index + 1]])
                            sbj_obj_point.append(
                                [sbj_entities_point[entity_index][0], obj_entities_point[entity_index + 1][0]])

                            sbj_obj.append([sbj_entities[entity_index + 1], obj_entities[entity_index]])
                            sbj_obj_point.append(
                                [sbj_entities_point[entity_index + 1][0], obj_entities_point[entity_index][0]])

                            entity_index += 2
                        else:
                            sbj_obj[-1][1] = obj_entities[entity_index]
                            sbj_obj_point[-1][1] = obj_entities_point[entity_index][0]

                            sbj_obj.append([sbj_entities[entity_index], obj_entities[entity_index - 1]])
                            sbj_obj_point.append(
                                [sbj_entities_point[entity_index][0], obj_entities_point[entity_index - 1][0]])
                            entity_index += 1
                    else:
                        sbj_obj.append([sbj_entities[entity_index], obj_entities[entity_index]])
                        sbj_obj_point.append([sbj_entities_point[entity_index][0], obj_entities_point[entity_index][0]])
                        entity_index += 1
            elif len(sbj_entities) >= 2 and len(obj_entities) >= 2:
                # 就近原则
                for subject_index in range(len(sbj_entities)):
                    nearest_object = []
                    nearest_object_point = []
                    nearest_distance = 999
                    for point in sbj_entities_point[subject_index]:
                        point_0 = (point[0] + point[1]) / 2
                        for object_index in range(len(obj_entities)):
                            for obj_point in obj_entities_point[object_index]:
                                point_1 = (obj_point[0] + obj_point[1]) / 2
                                distance = abs(point_0 - point_1)
                                if distance <= nearest_distance and point != obj_point:
                                    nearest_distance = distance
                                    nearest_object = obj_entities[object_index]
                                    nearest_object_point = obj_entities_point[object_index][0]
                    if nearest_object:
                        sbj_obj.append([sbj_entities[subject_index], nearest_object])
                        sbj_obj_point.append([sbj_entities_point[subject_index][0], nearest_object_point])
            return sbj_obj, sbj_obj_point

        def find_spo_by_complex(sbj_entities, obj_entities, sbj_entities_point, obj_entities_point, point_map,
                                mode='start'):
            sbj_obj = []
            sbj_obj_point = []
            if len(sbj_entities) == 1:
                for sbj_index, sbj_points in enumerate(sbj_entities_point):
                    for obj_index, obj_points in enumerate(obj_entities_point):
                        flag = False
                        for sbj_point_index, sbj_point in enumerate(sbj_points):
                            for obj_point_index, obj_point in enumerate(obj_points):
                                if sbj_point != obj_point:
                                    sbj_obj.append([sbj_entities[sbj_index], obj_entities[obj_index]])
                                    sbj_obj_point.append([sbj_entities_point[sbj_index][sbj_point_index],
                                                          obj_entities_point[obj_index][obj_point_index]])
                                    flag = True
                                    break
                            if flag:
                                break
            elif len(obj_entities) == 1:
                for obj_index, obj_points in enumerate(obj_entities_point):
                    for sbj_index, sbj_points in enumerate(sbj_entities_point):
                        flag = False
                        for obj_point_index, obj_point in enumerate(obj_points):
                            for sbj_point_index, sbj_point in enumerate(sbj_points):
                                if sbj_point != obj_point:
                                    sbj_obj.append([sbj_entities[sbj_index], obj_entities[obj_index]])
                                    sbj_obj_point.append([sbj_entities_point[sbj_index][sbj_point_index],
                                                          obj_entities_point[obj_index][obj_point_index]])
                                    flag = True
                                    break
                            if flag:
                                break
            elif len(sbj_entities) >= 2 and len(obj_entities) >= 2:
                sbj_obj, sbj_obj_point = find_spo_by_hp(sbj_entities, obj_entities, sbj_entities_point,
                                                        obj_entities_point, point_map, mode=mode)
            return sbj_obj, sbj_obj_point

        logits = preds[0]
        word_pred = preds[1]
        hot_point = preds[2]

        logits = logits > self.point_threshold
        logits = logits.cpu().numpy()

        word_pred = word_pred > self.point_threshold
        word_pred = word_pred.cpu().numpy()
        # hot_point = hot_point > self.point_threshold
        # hot_point = hot_point.cpu().numpy()

        spos = []
        spos_point = []
        for batch_index in range(logits.shape[0]):
            i_line = logits[batch_index][-1]
            # i_line = word_pred[batch_index]
            spos.append([])
            spos_point.append([])
            for schema_index in range(self.schema_num):
                sbj_line = logits[batch_index][schema_index]
                obj_line = logits[batch_index][schema_index + self.schema_num]
                subjects, subjects_point = find_entities(text[batch_index], sbj_line, i_line, mode='end')
                objects, objects_point = find_entities(text[batch_index], obj_line, i_line, mode='end')

                # sbj2obj, sbj2obj_point = find_spo_by_artificial_rules(subjects, objects, subjects_point, objects_point)
                sbj2obj, sbj2obj_point = find_spo_by_hp(subjects, objects, subjects_point, objects_point,
                                                        hot_point[batch_index], mode='end')
                # sbj2obj, sbj2obj_point = find_spo_by_complex(subjects, objects, subjects_point, objects_point,
                #                                              hot_point[batch_index])

                spos[batch_index].extend([[sbj, schema_index, obj] for sbj, obj in sbj2obj])
                spos_point[batch_index].extend([[sbj, schema_index, obj] for sbj, obj in sbj2obj_point])
        return spos, spos_point
