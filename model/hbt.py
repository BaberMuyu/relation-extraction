import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel

from global_config import BERT_MODEL, BIG_GPU
from modules import DGCNN
from utils import sl_loss


class HBTBase(nn.Module):
    def __init__(self, schema_num, sbj_type_num):
        super(HBTBase, self).__init__()
        self.schema_num = schema_num
        self.sbj_type_num = sbj_type_num
        self.point_threshold = 0.50
        self.start_num = 0
        self.end_num = 0

    def predicate_sbj(self, text_vec, text_mask):
        return 0

    def predicate_obj(self, text_vec, text_mask, text_select_indices, sbj_mask):
        return 0

    def forward(self, inputs):
        texts, texts_mask, text_select_indices, sbj_mask = inputs
        text_vec, word_vec = self.get_bert_vec(texts, texts_mask)
        text_vec = nn.functional.dropout(text_vec, p=0.1)
        # sbj
        sbj_points = self.predicate_sbj(text_vec, texts_mask)

        # obj
        obj_points = self.predicate_obj(text_vec, texts_mask, text_select_indices, sbj_mask)

        # word
        word_pred = self.word_linear(word_vec)
        word_pred = nn.functional.sigmoid(word_pred).squeeze(dim=-1)

        # scheam
        schema_vec = text_vec[:, 0, :]
        schema_pred = self.schema_linear(schema_vec)
        schema_pred = nn.functional.sigmoid(schema_pred).squeeze(dim=-1)
        return [sbj_points, obj_points, word_pred, schema_pred]

    def evaluate(self, inputs, raw_text):
        texts, texts_mask, text_select_indices, sbj_mask = inputs
        text_vec, word_vec = self.get_bert_vec(texts, texts_mask)

        # train part
        sbj_points = self.predicate_sbj(text_vec, texts_mask)
        obj_points = self.predicate_obj(text_vec, texts_mask, text_select_indices, sbj_mask)
        word_pred = nn.functional.sigmoid(self.word_linear(word_vec)).squeeze(dim=-1)
        schema_pred = nn.functional.sigmoid(self.schema_linear(text_vec[:, 0, :])).squeeze(dim=-1)

        # evaluate part
        sbj_entities, sbj_entities_point = self.find_sbj_entities(raw_text, sbj_points, word_pred)
        eval_sbj_mask, eval_text_select_indices = self.get_sbj_mask(sbj_entities_point, text_vec.size())
        eval_obj_points = self.predicate_obj(text_vec, texts_mask, eval_text_select_indices, eval_sbj_mask)
        obj_entites, obj_entities_point = self.find_obj_entities(raw_text, eval_obj_points, eval_text_select_indices,
                                                                 word_pred)

        spo_list, _ = self.get_spo_list(sbj_entities, obj_entites, sbj_entities_point, obj_entities_point)
        return [sbj_points, obj_points, word_pred, schema_pred], spo_list

    def predicate(self, inputs, raw_text):
        texts, texts_mask = inputs

        text_vec, word_vec = self.get_bert_vec(texts, texts_mask)

        # word
        word_pred = self.word_linear(word_vec)
        word_pred = nn.functional.sigmoid(word_pred).squeeze(dim=-1)

        # sbj points
        sbj_points = self.predicate_sbj(text_vec, texts_mask)

        # sbj mask, text_select_indices
        sbj_entities, sbj_entities_point = self.find_sbj_entities(raw_text, sbj_points, word_pred)
        sbjs_mask, text_select_indices = self.get_sbj_mask(sbj_entities_point, text_vec.size())

        # obj points
        obj_points = self.predicate_obj(text_vec, texts_mask, text_select_indices, sbjs_mask)
        obj_entites, obj_entities_point = self.find_obj_entities(raw_text, obj_points, text_select_indices, word_pred)

        # spo point list
        _, spo_point_list = self.get_spo_list(sbj_entities, obj_entites, sbj_entities_point, obj_entities_point)
        return spo_point_list

    def get_bert_vec(self, texts, texts_mask):
        _, _, text_vecs = self.bert(texts, texts_mask)
        text_vec = text_vecs[24] if BIG_GPU else text_vecs[12]
        word_vec = text_vecs[20] if BIG_GPU else text_vecs[10]
        if self.detach_bert:
            text_vec = text_vec.detach()
            word_vec = word_vec.detach()
        return text_vec, word_vec

    def get_sbj_mask(self, entities_point, input_shape):
        text_select_indices = []
        sbjs_mask = []
        for batch_index in range(input_shape[0]):
            for sbj_index in range(len(entities_point[batch_index])):
                sbj_mask = np.zeros(input_shape[1])
                for s, e in entities_point[batch_index][sbj_index]:
                    sbj_mask[s:e + 1] = 1
                sbjs_mask.append(sbj_mask)
                text_select_indices.append(batch_index)

        sbjs_mask = torch.tensor(sbjs_mask)
        text_select_indices = torch.tensor(text_select_indices)
        sbjs_mask = sbjs_mask.float().cuda()
        text_select_indices = text_select_indices.long().cuda()
        return sbjs_mask, text_select_indices

    def calculate_loss(self, preds, y_trues, mask, texts_select_indices):
        sbj_pred = torch.pow(preds[0], 1)
        obj_pred = torch.pow(preds[1], 1)
        word_pred = torch.pow(preds[2], 1)
        schema_pred = torch.pow(preds[3], 1)

        sbj_true = y_trues[0]
        obj_true = y_trues[1]
        word_true = y_trues[2]
        schema_true = y_trues[3]

        A = -8
        a = 1
        b = 1
        c = 1.7
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

        # schema loss
        schema_loss = sl_loss(schema_pred, schema_true, A, a, b)
        schema_loss = schema_loss.mean()

        loss = sbj_loss + obj_loss * 2.5 + word_loss * 0.015 + schema_loss * 0.01
        return loss

    # @classmethod
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
                    if pe_map[start_index] > pe_limit_map[start_index] and pe_limit_map[start_index] - start_index > 2:
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
                                                      obj_entities[batch_index][sbj_index][schema_index][obj_index]])
                        if sbj_entities_point is not None:
                            spo_point_list[batch_index].append([sbj_entities_point[batch_index][sbj_index],
                                                                schema_index,
                                                                obj_entities_point[batch_index][sbj_index][schema_index]
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


class HBT(HBTBase):
    def __init__(self, schema_num, sbj_type_num):
        super(HBT, self).__init__(schema_num, sbj_type_num)
        self.bert = BertModel.from_pretrained(BERT_MODEL, output_hidden_states=True)
        for p in self.bert.parameters():
            p.requires_grad = True

        feature_size = 1024 if BIG_GPU else 768
        self.sbj_linear = nn.Linear(feature_size, 128)
        self.sbj_linear_1 = nn.Linear(128, sbj_type_num * 2)
        self.obj_linear = nn.Linear(feature_size * 2, 512)
        # self.attention = EncoderLayer(model_dim=feature_size * 2, ffn_dim=feature_size * 2)
        self.obj_linear_1 = nn.Linear(512, schema_num * 2)
        self.schema_linear = nn.Linear(feature_size, schema_num)
        self.word_linear = nn.Linear(feature_size, 1)

        self.bert_parameters = [p for p in self.bert.parameters()]
        self.other_parameters = [p for p in self.sbj_linear.parameters()]
        self.other_parameters.extend([p for p in self.sbj_linear_1.parameters()])
        # self.other_parameters.extend([p for p in self.attention.parameters()])
        self.other_parameters.extend([p for p in self.obj_linear.parameters()])
        self.other_parameters.extend([p for p in self.obj_linear_1.parameters()])
        self.other_parameters.extend([p for p in self.schema_linear.parameters()])
        self.other_parameters.extend([p for p in self.word_linear.parameters()])
        self.detach_bert = True

    def predicate_sbj(self, text_vec, text_mask):
        sbj = nn.functional.relu(self.sbj_linear(text_vec))
        sbj = nn.functional.sigmoid(self.sbj_linear_1(sbj)).permute(0, 2, 1)
        return sbj

    def predicate_obj(self, text_vec, text_mask, text_select_indices, sbj_mask):
        text_vec = torch.index_select(text_vec, 0, text_select_indices)
        sbj_vec = (text_vec * sbj_mask.unsqueeze(dim=-1)).sum(dim=1)
        sbj_mask_sum = sbj_mask.sum(dim=1)
        sbj_mask_sum += sbj_mask_sum.eq(0).float()  # 防止除零
        sbj_vec = sbj_vec / sbj_mask_sum.unsqueeze(dim=-1)

        sbj2obj_vec = torch.cat([text_vec, sbj_vec.unsqueeze(dim=1).repeat(1, text_vec.size()[1], 1)], dim=-1)
        obj = nn.functional.relu(self.obj_linear(sbj2obj_vec))
        obj = nn.functional.sigmoid(self.obj_linear_1(obj)).permute(0, 2, 1)
        return obj


class DGCNN_HBT(HBTBase):
    def __init__(self, schema_num, sbj_type_num, enable_hands=False):
        super(DGCNN_HBT, self).__init__(schema_num, sbj_type_num)
        self.enable_hands = enable_hands
        self.bert = BertModel.from_pretrained(BERT_MODEL, output_hidden_states=True)
        for p in self.bert.parameters():
            p.requires_grad = True
        feature_size = 1024 if BIG_GPU else 768

        if enable_hands:
            self.sbj_linear = nn.Linear(feature_size + 301, 256)
        else:
            self.sbj_linear = nn.Linear(feature_size + 300, 256)
        sbj_dgcnn_settings = [
            {'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'dilation': 2},
            {'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'dilation': 5},
            {'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'dilation': 11},
        ]
        self.sbj_dgcnn = DGCNN(sbj_dgcnn_settings)
        self.sbj_linear_1 = nn.Linear(256, sbj_type_num * 2)

        if enable_hands:
            self.obj_linear = nn.Linear((feature_size + 301) * 2, 512)
        else:
            self.obj_linear = nn.Linear((feature_size + 300) * 2, 512)
        obj_dgcnn_settings = [
            {'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'dilation': 2},
            {'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'dilation': 5},
            {'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'dilation': 11},
        ]
        self.obj_dgcnn = DGCNN(obj_dgcnn_settings)
        self.obj_linear_1 = nn.Linear(512, schema_num * 2)

        self.schema_linear = nn.Linear(feature_size, schema_num)
        self.word_linear = nn.Linear(feature_size, 1)

        self.bert_parameters = [p for p in self.bert.parameters()]
        self.other_parameters = [p for p in self.sbj_dgcnn.parameters()]
        self.other_parameters.extend([p for p in self.sbj_linear_1.parameters()])
        self.other_parameters.extend([p for p in self.sbj_linear.parameters()])
        self.other_parameters.extend([p for p in self.obj_dgcnn.parameters()])
        self.other_parameters.extend([p for p in self.obj_linear_1.parameters()])
        self.other_parameters.extend([p for p in self.schema_linear.parameters()])
        self.other_parameters.extend([p for p in self.word_linear.parameters()])
        self.detach_bert = True

    def predicate_sbj(self, text_vec, text_mask):
        sbj = self.sbj_linear(text_vec)
        sbj = self.sbj_dgcnn(sbj, text_mask)
        sbj = nn.functional.sigmoid(self.sbj_linear_1(sbj)).permute(0, 2, 1)
        return sbj

    def predicate_obj(self, text_vec, text_mask, text_select_indices, sbj_mask):
        text_vec = torch.index_select(text_vec, 0, text_select_indices)
        text_mask = torch.index_select(text_mask, 0, text_select_indices)
        sbj_vec = (text_vec * sbj_mask.unsqueeze(dim=-1)).sum(dim=1)
        sbj_mask_sum = sbj_mask.sum(dim=1)
        sbj_mask_sum += sbj_mask_sum.eq(0).float()  # 防止除零
        sbj_vec = sbj_vec / sbj_mask_sum.unsqueeze(dim=-1)

        obj = torch.cat([text_vec, sbj_vec.unsqueeze(dim=1).repeat(1, text_vec.size()[1], 1)], dim=-1)
        obj = self.obj_linear(obj)
        obj = self.obj_dgcnn(obj, text_mask)
        obj = nn.functional.sigmoid(self.obj_linear_1(obj)).permute(0, 2, 1)
        return obj

    def forward(self, inputs):
        texts, texts_mask, text_select_indices, sbj_mask, w2vs, hands = inputs
        text_vec, word_vec = self.get_bert_vec(texts, texts_mask)
        schema_vec = text_vec[:, 0, :]
        if self.enable_hands:
            text_vec = torch.cat([text_vec, w2vs, hands.unsqueeze(dim=-1)], dim=-1)
        else:
            text_vec = torch.cat([text_vec, w2vs], dim=-1)
        #
        text_vec = nn.functional.dropout(text_vec, p=0.1)
        # sbj
        sbj_points = self.predicate_sbj(text_vec, texts_mask)

        # obj
        obj_points = self.predicate_obj(text_vec, texts_mask, text_select_indices, sbj_mask)

        # word
        word_pred = self.word_linear(word_vec)
        word_pred = nn.functional.sigmoid(word_pred).squeeze(dim=-1)

        # scheam
        schema_pred = self.schema_linear(schema_vec)
        schema_pred = nn.functional.sigmoid(schema_pred).squeeze(dim=-1)
        return [sbj_points, obj_points, word_pred, schema_pred]

    def evaluate(self, inputs, raw_text):
        texts, texts_mask, text_select_indices, sbj_mask, w2vs, hands = inputs
        text_vec, word_vec = self.get_bert_vec(texts, texts_mask)
        schema_vec = text_vec[:, 0, :]
        if self.enable_hands:
            text_vec = torch.cat([text_vec, w2vs, hands.unsqueeze(dim=-1)], dim=-1)
        else:
            text_vec = torch.cat([text_vec, w2vs], dim=-1)

        # train part
        sbj_points = self.predicate_sbj(text_vec, texts_mask)
        obj_points = self.predicate_obj(text_vec, texts_mask, text_select_indices, sbj_mask)
        word_pred = nn.functional.sigmoid(self.word_linear(word_vec)).squeeze(dim=-1)
        schema_pred = nn.functional.sigmoid(self.schema_linear(schema_vec)).squeeze(dim=-1)

        # evaluate part
        sbj_entities, sbj_entities_point = self.find_sbj_entities(raw_text, sbj_points, word_pred)
        eval_sbj_mask, eval_text_select_indices = self.get_sbj_mask(sbj_entities_point, text_vec.size())
        eval_obj_points = self.predicate_obj(text_vec, texts_mask, eval_text_select_indices, eval_sbj_mask)
        obj_entites, obj_entities_point = self.find_obj_entities(raw_text, eval_obj_points, eval_text_select_indices,
                                                                 word_pred)

        spo_list, _ = self.get_spo_list(sbj_entities, obj_entites, sbj_entities_point, obj_entities_point)
        return [sbj_points, obj_points, word_pred, schema_pred], spo_list

    def predicate(self, inputs, raw_text):
        texts, texts_mask, w2vs, hands = inputs

        text_vec, word_vec = self.get_bert_vec(texts, texts_mask)
        if self.enable_hands:
            text_vec = torch.cat([text_vec, w2vs, hands.unsqueeze(dim=-1)], dim=-1)
        else:
            text_vec = torch.cat([text_vec, w2vs], dim=-1)

        # word
        word_pred = self.word_linear(word_vec)
        word_pred = nn.functional.sigmoid(word_pred).squeeze(dim=-1)

        # sbj points
        sbj_points = self.predicate_sbj(text_vec, texts_mask)

        # sbj mask, text_select_indices
        sbj_entities, sbj_entities_point = self.find_sbj_entities(raw_text, sbj_points, word_pred)
        sbjs_mask, text_select_indices = self.get_sbj_mask(sbj_entities_point, text_vec.size())

        # obj points
        obj_points = self.predicate_obj(text_vec, texts_mask, text_select_indices, sbjs_mask)
        obj_entites, obj_entities_point = self.find_obj_entities(raw_text, obj_points, text_select_indices, word_pred)

        # spo point list
        _, spo_point_list = self.get_spo_list(sbj_entities, obj_entites, sbj_entities_point, obj_entities_point)
        return spo_point_list

    def predicate_1(self, inputs):
        texts, texts_mask, w2vs, hands = inputs

        text_vec, word_vec = self.get_bert_vec(texts, texts_mask)
        if self.enable_hands:
            text_vec = torch.cat([text_vec, w2vs, hands.unsqueeze(dim=-1)], dim=-1)
        else:
            text_vec = torch.cat([text_vec, w2vs], dim=-1)

        # word
        word_pred = self.word_linear(word_vec)
        word_pred = nn.functional.sigmoid(word_pred).squeeze(dim=-1)
        sbj_points = self.predicate_sbj(text_vec, texts_mask)
        return text_vec, word_pred, sbj_points

    def predicate_1_1(self, text_vec, sbj_points, word_pred, raw_text):
        sbj_entities, sbj_entities_point = self.find_sbj_entities(raw_text, sbj_points, word_pred)
        sbjs_mask, text_select_indices = self.get_sbj_mask(sbj_entities_point, text_vec.size())
        return sbjs_mask, text_select_indices, sbj_entities, sbj_entities_point

    def predicate_2(self, text_vec, texts_mask, text_select_indices, sbjs_mask):
        obj_points = self.predicate_obj(text_vec, texts_mask, text_select_indices, sbjs_mask)
        return obj_points

    def predicate_2_2(self, obj_points, text_select_indices, word_pred, sbj_entities, sbj_entities_point, raw_text):
        obj_entites, obj_entities_point = self.find_obj_entities(raw_text, obj_points, text_select_indices, word_pred)

        # spo point list
        _, spo_point_list = self.get_spo_list(sbj_entities, obj_entites, sbj_entities_point, obj_entities_point)
        return spo_point_list


class RNN_HBT(HBTBase):
    def __init__(self, schema_num, sbj_type_num):
        super(RNN_HBT, self).__init__(schema_num, sbj_type_num)
        self.bert = BertModel.from_pretrained(BERT_MODEL, output_hidden_states=True)
        for p in self.bert.parameters():
            p.requires_grad = True
        feature_size = 1024 if BIG_GPU else 768

        self.sbj_linear = nn.Linear(feature_size + 301, 128)
        self.sbj_gru = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.sbj_linear_1 = nn.Linear(128, sbj_type_num * 2)

        self.obj_linear = nn.Linear((feature_size + 301) * 2, 256)
        self.obj_gru = nn.GRU(256, 128, batch_first=True, bidirectional=True)
        self.obj_linear_1 = nn.Linear(256, schema_num * 2)

        self.schema_linear = nn.Linear(feature_size, schema_num)
        self.word_linear = nn.Linear(feature_size, 1)

        self.bert_parameters = [p for p in self.bert.parameters()]
        self.other_parameters = [p for p in self.sbj_gru.parameters()]
        self.other_parameters.extend([p for p in self.sbj_linear_1.parameters()])
        self.other_parameters.extend([p for p in self.sbj_linear.parameters()])
        self.other_parameters.extend([p for p in self.obj_linear.parameters()])
        self.other_parameters.extend([p for p in self.obj_gru.parameters()])
        self.other_parameters.extend([p for p in self.obj_linear_1.parameters()])
        self.other_parameters.extend([p for p in self.schema_linear.parameters()])
        self.other_parameters.extend([p for p in self.word_linear.parameters()])
        self.detach_bert = True

    def predicate_sbj(self, text_vec, text_mask):
        sequence_length = text_mask.sum(dim=-1)
        res = nn.functional.tanh(self.sbj_linear(text_vec))
        res = res * text_mask.unsqueeze(dim=-1)
        sbj = torch.nn.utils.rnn.pack_padded_sequence(res, sequence_length, batch_first=True, enforce_sorted=False)
        sbj, _ = self.sbj_gru(sbj)
        sbj, _ = torch.nn.utils.rnn.pad_packed_sequence(sbj, batch_first=True)
        sbj = res + sbj
        sbj = nn.functional.sigmoid(self.sbj_linear_1(sbj)).permute(0, 2, 1)
        return sbj

    def predicate_obj(self, text_vec, text_mask, text_select_indices, sbj_mask):
        text_vec = torch.index_select(text_vec, 0, text_select_indices)
        text_mask = torch.index_select(text_mask, 0, text_select_indices)
        sbj_vec = (text_vec * sbj_mask.unsqueeze(dim=-1)).sum(dim=1)
        sbj_mask_sum = sbj_mask.sum(dim=1)
        sbj_mask_sum += sbj_mask_sum.eq(0).float()  # 防止除零
        sbj_vec = sbj_vec / sbj_mask_sum.unsqueeze(dim=-1)

        sbj2obj_vec = torch.cat([text_vec, sbj_vec.unsqueeze(dim=1).repeat(1, text_vec.size()[1], 1)], dim=-1)
        # obj, _ = self.attention(sbj2obj_vec, text_mask)
        res = nn.functional.tanh(self.obj_linear(sbj2obj_vec))
        res = res * text_mask.unsqueeze(dim=-1)
        sequence_length = text_mask.sum(dim=-1)
        obj = torch.nn.utils.rnn.pack_padded_sequence(res, sequence_length, batch_first=True, enforce_sorted=False)
        obj, _ = self.obj_gru(obj)
        obj, _ = torch.nn.utils.rnn.pad_packed_sequence(obj, batch_first=True, total_length=text_mask.size(1))
        obj = obj + res
        obj = nn.functional.sigmoid(self.obj_linear_1(obj)).permute(0, 2, 1)
        return obj

    def forward(self, inputs):
        texts, texts_mask, text_select_indices, sbj_mask, w2vs, hands = inputs
        text_vec, word_vec = self.get_bert_vec(texts, texts_mask)
        schema_vec = text_vec[:, 0, :]
        text_vec = torch.cat([text_vec, w2vs, hands.unsqueeze(dim=-1)], dim=-1)
        # text_vec = torch.cat([text_vec, w2vs, hands.unsqueeze(dim=-1)], dim=-1)
        # text_vec = nn.functional.dropout(text_vec, p=0.1)
        text_vec = nn.functional.dropout2d(text_vec.permute(0, 2, 1).squeeze(dim=-1), 0.2)
        text_vec = text_vec.squeeze(dim=-1).permute(0, 2, 1)
        # sbj
        sbj_points = self.predicate_sbj(text_vec, texts_mask)

        # obj
        obj_points = self.predicate_obj(text_vec, texts_mask, text_select_indices, sbj_mask)

        # word
        word_pred = self.word_linear(word_vec)
        word_pred = nn.functional.sigmoid(word_pred).squeeze(dim=-1)

        # scheam
        schema_pred = self.schema_linear(schema_vec)
        schema_pred = nn.functional.sigmoid(schema_pred).squeeze(dim=-1)
        return [sbj_points, obj_points, word_pred, schema_pred]

    def evaluate(self, inputs, raw_text):
        texts, texts_mask, text_select_indices, sbj_mask, w2vs, hands = inputs
        text_vec, word_vec = self.get_bert_vec(texts, texts_mask)
        schema_vec = text_vec[:, 0, :]
        text_vec = torch.cat([text_vec, w2vs, hands.unsqueeze(dim=-1)], dim=-1)
        # text_vec = torch.cat([text_vec, w2vs, hands.unsqueeze(dim=-1)], dim=-1)

        # train part
        sbj_points = self.predicate_sbj(text_vec, texts_mask)
        obj_points = self.predicate_obj(text_vec, texts_mask, text_select_indices, sbj_mask)
        word_pred = nn.functional.sigmoid(self.word_linear(word_vec)).squeeze(dim=-1)
        schema_pred = nn.functional.sigmoid(self.schema_linear(schema_vec)).squeeze(dim=-1)

        # evaluate part
        sbj_entities, sbj_entities_point = self.find_sbj_entities(raw_text, sbj_points, word_pred)
        eval_sbj_mask, eval_text_select_indices = self.get_sbj_mask(sbj_entities_point, text_vec.size())
        eval_obj_points = self.predicate_obj(text_vec, texts_mask, eval_text_select_indices, eval_sbj_mask)
        obj_entites, obj_entities_point = self.find_obj_entities(raw_text, eval_obj_points, eval_text_select_indices,
                                                                 word_pred)

        spo_list, _ = self.get_spo_list(sbj_entities, obj_entites, sbj_entities_point, obj_entities_point)
        return [sbj_points, obj_points, word_pred, schema_pred], spo_list

    def predicate(self, inputs, raw_text):
        texts, texts_mask, w2vs, hands = inputs

        text_vec, word_vec = self.get_bert_vec(texts, texts_mask)
        text_vec = torch.cat([text_vec, w2vs, hands.unsqueeze(dim=-1)], dim=-1)
        # text_vec = torch.cat([text_vec, w2vs, hands.unsqueeze(dim=-1)], dim=-1)

        # word
        word_pred = self.word_linear(word_vec)
        word_pred = nn.functional.sigmoid(word_pred).squeeze(dim=-1)

        # sbj points
        sbj_points = self.predicate_sbj(text_vec, texts_mask)

        # sbj mask, text_select_indices
        sbj_entities, sbj_entities_point = self.find_sbj_entities(raw_text, sbj_points, word_pred)
        sbjs_mask, text_select_indices = self.get_sbj_mask(sbj_entities_point, text_vec.size())

        # obj points
        obj_points = self.predicate_obj(text_vec, texts_mask, text_select_indices, sbjs_mask)
        obj_entites, obj_entities_point = self.find_obj_entities(raw_text, obj_points, text_select_indices, word_pred)

        # spo point list
        _, spo_point_list = self.get_spo_list(sbj_entities, obj_entites, sbj_entities_point, obj_entities_point)
        return spo_point_list

    def predicate_1(self, inputs):
        texts, texts_mask, w2vs, hands = inputs

        text_vec, word_vec = self.get_bert_vec(texts, texts_mask)
        text_vec = torch.cat([text_vec, w2vs, hands.unsqueeze(dim=-1)], dim=-1)

        # word
        word_pred = self.word_linear(word_vec)
        word_pred = nn.functional.sigmoid(word_pred).squeeze(dim=-1)
        sbj_points = self.predicate_sbj(text_vec, texts_mask)
        return text_vec, word_pred, sbj_points

    def predicate_1_1(self, text_vec, sbj_points, word_pred, raw_text):
        sbj_entities, sbj_entities_point = self.find_sbj_entities(raw_text, sbj_points, word_pred)
        sbjs_mask, text_select_indices = self.get_sbj_mask(sbj_entities_point, text_vec.size())
        return sbjs_mask, text_select_indices, sbj_entities, sbj_entities_point

    def predicate_2(self, text_vec, texts_mask, text_select_indices, sbjs_mask):
        obj_points = self.predicate_obj(text_vec, texts_mask, text_select_indices, sbjs_mask)
        return obj_points

    def predicate_2_2(self, obj_points, text_select_indices, word_pred, sbj_entities, sbj_entities_point, raw_text):
        obj_entites, obj_entities_point = self.find_obj_entities(raw_text, obj_points, text_select_indices, word_pred)

        # spo point list
        _, spo_point_list = self.get_spo_list(sbj_entities, obj_entites, sbj_entities_point, obj_entities_point)
        return spo_point_list

