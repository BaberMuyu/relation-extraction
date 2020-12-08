from torch.utils.data import Dataset

from utils import MyBertTokenizer
import numpy as np
import re
import copy
import pickle, pkuseg, json
import os

from global_config import ROOT_DATA, TEST_MODE, BERT_MODEL, ROOT_RESULT, CONVERT_DATA, POSTPROCESS
from utils.metrics import calculate_f1
from utils import kmp, KnowledgeGraph, std_kg, is_chinese, have_chinese


# max length 定为205 包括cls 和 seq
# 实际max length 300
class RawDataSet(Dataset):
    def __init__(self, file_name):
        file_name = ROOT_DATA + 'lic_2020/' + file_name
        with open(file_name, mode='r', encoding='utf-8') as fp:
            self.data = json.load(fp)
        if TEST_MODE:
            self.len = 2000
        else:
            self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index]


class LIC2020Data(object):
    data_set = {"train": "train_tokenizer.json",
                'dev': 'dev_tokenizer.json',
                'test1': 'test1_tokenizer.json',
                'test2': 'test2_tokenizer.json'}

    def __new__(cls, mode):
        if mode in cls.data_set.keys():
            return RawDataSet(cls.data_set[mode])
        elif mode == 'schema':
            return LIC2020Schema()
        else:
            assert 0


class LIC2020Schema(object):
    def __init__(self):
        # raw schema
        file_name = ROOT_DATA + 'lic_2020/schema.json'
        with open(file_name, mode='r', encoding='utf-8') as fp:
            data = fp.readlines()
            self.data = [json.loads(d) for d in data]
        self.schema_dict = dict((d['predicate'], d) for d in self.data)
        self.raw_schema_predicate_list = [d['predicate'] for d in self.data]

        # my schema
        file_name = ROOT_DATA + 'lic_2020/my_schema.json'
        with open(file_name, mode='r', encoding='utf-8') as fp:
            data = fp.readlines()
            self.data = [json.loads(d) for d in data]

        self.expand_schema_dict = dict((d['predicate'], d) for d in self.data)
        self.predicate_list = [d['predicate'] for d in self.data]
        self.subject_type_list = ['图书作品', '企业/品牌', '学校', '歌曲', '行政区', '文学作品', '景点', 'Number', '奖项',
                                  '人物', '国家', '历史人物', '地点', '电视综艺', '机构', '娱乐人物', '学科专业', '作品',
                                  '影视作品', '企业']
        self.object_type_list = ['学校', '歌曲', '气候', '奖项', 'Text', '音乐专辑', '人物', '国家', '企业', '地点', '作品',
                                 '城市', '语言', '影视作品', 'Date', 'Number']

        self.pred2id_dict = dict((pred, i) for i, pred in enumerate(self.predicate_list))
        self.id2pred_dict = dict((i, pred) for i, pred in enumerate(self.predicate_list))

        self.obj2id_dict = dict((obj, i) for i, obj in enumerate(self.object_type_list))
        self.id2obj_dict = dict((i, obj) for i, obj in enumerate(self.object_type_list))

        self.sbj2id_dict = dict((sbj, i) for i, sbj in enumerate(self.subject_type_list))
        self.id2sbj_dict = dict((i, sbj) for i, sbj in enumerate(self.subject_type_list))


s = 0


def locate_entity(text, ojbk, ojbk_points):
    change_flag = False
    for s, e in ojbk_points:
        if text[s:e + 1] != ojbk:
            change_flag = True
            break
    if not change_flag:
        return ojbk_points
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


def postprocess(input_text, input_spos):
    return input_spos


def postprocess_1(input_text, input_spos):
    def add_period(text, spo):
        if spo['predicate'] == '获奖':
            jie_index = text.find(spo['object']['@value']) - 1
            while True:
                if text[jie_index] == ' ':
                    jie_index -= 1
                else:
                    break
            if text[jie_index] == '届' or text[jie_index] == '回':
                period = ''
                warning_index = -1
                for index in range(jie_index - 1, -1, -1):
                    if text[index] in '0123456789一二三四五六七八九十零首':
                        if text[index] == '首':
                            period = text[index]  # need to test 首 or 首届
                            break
                        else:
                            period = text[index] + period
                    else:
                        warning_index = index
                        break
                if period:
                    spo['object_type']['period'] = 'Number'
                    spo['object']['period'] = period
                else:
                    print(text)
                    print('period warning', text[warning_index])
        return spo

    def home_relation(spos):
        families = {}
        new_spos = []
        for spo in spos:
            if spo['predicate'] == '父亲':
                if spo['subject'] not in families.keys():
                    families[spo['subject']] = {'father': spo['object']['@value'], 'mother': ''}
                else:
                    families[spo['subject']]['father'] = spo['object']['@value']
            elif spo['predicate'] == '母亲':
                if spo['subject'] not in families.keys():
                    families[spo['subject']] = {'father': '', 'mother': spo['object']['@value']}
                else:
                    families[spo['subject']]['mother'] = spo['object']['@value']

            if spo['predicate'] == '妻子':
                new_spos.append({'subject_type': '人物', 'subject': spo['object']['@value'], 'predicate': '丈夫',
                                 'object_type': {'@value': '人物'}, 'object': {'@value': spo['subject']}})

            if spo['predicate'] == '丈夫':
                continue

        for son in families.keys():
            if families[son]['mother'] and families[son]['father'] and families[son]['mother'] != families[son][
                'father']:
                new_spos.append({'subject_type': '人物', 'subject': families[son]['mother'], 'predicate': '丈夫',
                                 'object_type': {'@value': '人物'}, 'object': {'@value': families[son]['father']}})
                new_spos.append({'subject_type': '人物', 'subject': families[son]['father'], 'predicate': '妻子',
                                 'object_type': {'@value': '人物'}, 'object': {'@value': families[son]['mother']}})
        for spo in new_spos:
            if spo not in spos:
                spos.append(spo)
        return spos

    for spo in input_spos[:]:
        spo_index = input_spos.index(spo)
        spo = add_period(input_text, spo)
        input_spos[spo_index] = spo
    input_spos = home_relation(input_spos)
    out_spos = []
    for spo in input_spos:
        a = {'subject_type': spo['subject_type'],
             'subject': spo['subject'],
             'predicate': spo['predicate'],
             'object_type': spo['object_type'],
             'object': spo['object']}
        if a not in out_spos:
            out_spos.append(a)
    return out_spos


def get_index_map(text, tokenizer):
    text_encoded = tokenizer.encode(text)
    text_decoded = tokenizer.special_decode(text_encoded)

    decode2raw_map = [0] * len(text_decoded)
    decode2raw_map[-1] = len(text)
    text_index = 0
    decode_index = 1

    while True:
        sub_str = text[text_index]
        sub_str_encode = tokenizer.encode(sub_str, add_special_tokens=False)
        if sub_str == text_decoded[decode_index] == ' ':
            decode2raw_map[decode_index] = text_index
            decode_index += 1
            text_index += 1
        elif sub_str == ' ' and text_decoded[decode_index] == '  ':
            decode2raw_map[decode_index] = text_index
            decode_index += 1
            text_index += 2
        elif not sub_str_encode:
            text_index += 1
        elif text_decoded[decode_index] == '[UNK]':  # "多对一"
            unknow_start = text_index
            unknow_end = text_index
            while True:
                if unknow_end == len(text) - 1:
                    decode2raw_map[decode_index] = unknow_start
                    text_index = unknow_end + 1
                    decode_index += 1
                    break
                unknow_end += 1
                sub_str = text[unknow_start: unknow_end + 1]
                sub_str_encode = tokenizer.encode(sub_str, add_special_tokens=False)
                if len(sub_str_encode) >= 2 and sub_str_encode[0] == 100:
                    decode2raw_map[decode_index] = unknow_start
                    text_index = unknow_end if text[unknow_end - 1] != ' ' else (unknow_end - 1)
                    decode_index += 1
                    break
        elif len(sub_str_encode) > 1:  # 一对多
            for iii in range(len(sub_str_encode)):
                decode2raw_map[decode_index] = text_index
                decode_index += 1
            text_index += 1
        elif len(text_decoded[decode_index]) == 1:
            decode2raw_map[decode_index] = text_index
            text_index += 1
            decode_index += 1
        elif len(text_decoded[decode_index]) > 1:
            sub_len = 2
            while True:
                sub_str_encode = tokenizer.encode(text[text_index:text_index + sub_len], add_special_tokens=False)
                sub_str_decode = tokenizer.special_decode(sub_str_encode)
                sub_str_decode = ''.join(sub_str_decode)
                if sub_str_decode == text_decoded[decode_index]:
                    break
                elif len(sub_str_decode) > len(text_decoded[decode_index]):
                    print('error')
                    break
                else:
                    sub_len += 1
            decode2raw_map[decode_index] = text_index
            text_index += sub_len
            decode_index += 1
        else:
            print('error: generate map index')
            text_index += 1
        if text_index >= len(text):
            break
    return decode2raw_map


def get_formal_result(text, spos, tokenizer, schema, post=True):
    point_map = get_index_map(text, tokenizer)
    formal_spos = []
    for spo in spos:
        sbj = text[point_map[spo[0][0][0]]: point_map[spo[0][0][1] + 1]].strip()
        obj = text[point_map[spo[2][0][0]]: point_map[spo[2][0][1] + 1]].strip()
        formal_spos.append({
            'subject_type': schema.expand_schema_dict[schema.id2pred_dict[spo[1]]]['subject_type'],
            'subject': sbj,
            'subject_point': [(point_map[s], point_map[s] + len(sbj) - 1) for s, e in spo[0]],
            'predicate': schema.id2pred_dict[spo[1]],
            'object_type': {'@value': schema.expand_schema_dict[schema.id2pred_dict[spo[1]]]['object_type']['@value']},
            'object': {'@value': obj},
            'object_point': {'@value': [(point_map[s], point_map[s] + len(obj) - 1) for s, e in spo[2]]}})

    if post:
        formal_spos = postprocess(text, formal_spos)
        formal_spos = combine_spos(formal_spos)
        formal_spos = postprocess_1(text, formal_spos)

    pair = {'text': text, 'spo_list': formal_spos}
    return pair


def generate_formal_results(spo_list, data_type, output_file):
    tokenizer = MyBertTokenizer.from_pretrained(BERT_MODEL)

    data_set = {"train": "clean_train_data.json",
                'dev': 'clean_dev_data.json',
                'test1': 'test1_data.json',
                'test2': 'test2_data.json'}
    schema = LIC2020Schema()
    fp = open(ROOT_DATA + 'lic_2020/' + data_set[data_type], 'r', encoding='utf-8')
    if data_type in('test1', 'test2'):
        data = fp.readlines()
    else:
        data = json.load(fp)
    out_fp = open(ROOT_RESULT + output_file, 'w+', encoding='utf-8')

    for pair, spos in zip(data, spo_list):
        if data_type in('test1', 'test2'):
            pair = json.loads(pair)
        new_pair = get_formal_result(pair['text'], spos, tokenizer, schema, post=POSTPROCESS)
        string = json.dumps(new_pair, ensure_ascii=False) + '\n'
        out_fp.write(string)


# ======================================================================================================================
def expand_spo(spo):
    def dub(spo):
        expand_spo_list = []
        spo_temp = {'predicate': '配音', 'object_type': {'@value': '人物'}, 'subject_type': '娱乐人物',
                    'object': {'@value': spo['object']['@value']}, 'subject': spo['subject']}
        expand_spo_list.append(spo_temp)
        if 'inWork' in spo['object'].keys():
            spo_temp = {'predicate': '角色', 'object_type': {'@value': '人物'}, 'subject_type': '影视作品',
                        'object': {'@value': spo['object']['@value']}, 'subject': spo['object']['inWork']}
            expand_spo_list.append(spo_temp)
        return expand_spo_list

    def release_date(spo):
        expand_spo_list = []
        spo_temp = {'predicate': '上映时间', 'object_type': {'@value': 'Date'}, 'subject_type': '影视作品',
                    'object': {'@value': spo['object']['@value']}, 'subject': spo['subject']}
        expand_spo_list.append(spo_temp)
        if 'inArea' in spo['object'].keys():
            spo_temp = {'predicate': '上映地点-时间', 'object_type': {'@value': 'Date'}, 'subject_type': '地点',
                        'object': {'@value': spo['object']['@value']}, 'subject': spo['object']['inArea']}
            expand_spo_list.append(spo_temp)
        return expand_spo_list

    def box_office(spo):
        expand_spo_list = []
        spo_temp = {'predicate': '票房', 'object_type': {'@value': 'Number'}, 'subject_type': '影视作品',
                    'object': {'@value': spo['object']['@value']}, 'subject': spo['subject']}
        expand_spo_list.append(spo_temp)
        if 'inArea' in spo['object'].keys():
            spo_temp = {'predicate': '票房区域', 'object_type': {'@value': '地点'}, 'subject_type': 'Number',
                        'object': {'@value': spo['object']['inArea']}, 'subject': spo['object']['@value']}
            expand_spo_list.append(spo_temp)
        return expand_spo_list

    def win_a_prize(spo):
        expand_spo_list = []
        spo_temp = {'predicate': '获奖', 'object_type': {'@value': '奖项'}, 'subject_type': '娱乐人物',
                    'object': {'@value': spo['object']['@value']}, 'subject': spo['subject']}
        expand_spo_list.append(spo_temp)
        if 'inWork' in spo['object'].keys():
            spo_temp = {'predicate': '作品', 'object_type': {'@value': '作品'}, 'subject_type': '娱乐人物',
                        'object': {'@value': spo['object']['inWork']}, 'subject': spo['subject']}
            expand_spo_list.append(spo_temp)
            spo_temp = {'predicate': '作品获奖', 'object_type': {'@value': '奖项'}, 'subject_type': '作品',
                        'object': {'@value': spo['object']['@value']}, 'subject': spo['object']['inWork']}
            expand_spo_list.append(spo_temp)

        if 'onDate' in spo['object'].keys():
            spo_temp = {'predicate': '奖项时间', 'object_type': {'@value': 'Date'}, 'subject_type': '奖项',
                        'object': {'@value': spo['object']['onDate']}, 'subject': spo['object']['@value']}
            expand_spo_list.append(spo_temp)

        ### 删除了届！！！！
        return expand_spo_list

    def portray(spo):
        expand_spo_list = []
        spo_temp = {'predicate': '饰演', 'object_type': {'@value': '人物'}, 'subject_type': '娱乐人物',
                    'object': {'@value': spo['object']['@value']}, 'subject': spo['subject']}
        expand_spo_list.append(spo_temp)
        if 'inWork' in spo['object'].keys():
            spo_temp = {'predicate': '角色', 'object_type': {'@value': '人物'}, 'subject_type': '影视作品',
                        'object': {'@value': spo['object']['@value']}, 'subject': spo['object']['inWork']}
            expand_spo_list.append(spo_temp)
            spo_temp = {'predicate': '参与影视作品', 'object_type': {'@value': '影视作品'}, 'subject_type': '人物',
                        'object': {'@value': spo['object']['inWork']}, 'subject': spo['subject']}
            expand_spo_list.append(spo_temp)
        return expand_spo_list

    complex_scheams = {'配音': dub,
                       '上映时间': release_date,
                       '票房': box_office,
                       '获奖': win_a_prize,
                       '饰演': portray}
    if spo['predicate'] in complex_scheams.keys():
        expand_list = complex_scheams[spo['predicate']](spo)
    else:
        expand_list = [spo]
    return expand_list


def combine_spos(spos_inputs):
    def dub(spos):
        complex_spos = []
        while True:
            try:
                predicate_list = [spo['predicate'] for spo in spos]
                pred_index = predicate_list.index('配音')
                complex_spo = spos[pred_index]
                flag = False
                for spo in spos:  # 同一个人物也有可能在不同影视作品 其实也不合理
                    if spo['predicate'] == '角色' and spo['object']['@value'] == spos[pred_index]['object']['@value']:
                        complex_spo = {"object_type": {"inWork": "影视作品", "@value": "人物"},
                                       "predicate": '配音',
                                       "object": {'inWork': spo['subject'], '@value': spo['object']['@value']},
                                       "subject_type": "娱乐人物",
                                       "subject": spos[pred_index]['subject']}
                        complex_spos.append(complex_spo)
                        flag = True
                if not flag:
                    complex_spos.append(complex_spo)
                del (spos[pred_index])
            except ValueError:
                break
        return complex_spos

    def release_date(spos):
        complex_spos = []
        while True:
            try:
                predicate_list = [spo['predicate'] for spo in spos]
                pred_index = predicate_list.index('上映时间')
                complex_spo = spos[pred_index]
                for spo in spos:
                    if spo['predicate'] == '上映地点-时间' and \
                            spo['object']['@value'] == spos[pred_index]['object']['@value']:
                        complex_spo = {"object_type": {"inArea": "地点", "@value": "Date"},
                                       "predicate": '上映时间',
                                       "object": {'inArea': spo['subject'],
                                                  '@value': spos[pred_index]['object']['@value']},
                                       "subject_type": "影视作品",
                                       "subject": spos[pred_index]['subject']}

                        break
                complex_spos.append(complex_spo)
                del (spos[pred_index])
            except ValueError:
                break
        return complex_spos

    def box_office(spos):
        complex_spos = []
        while True:
            try:
                predicate_list = [spo['predicate'] for spo in spos]
                pred_index = predicate_list.index('票房')
                complex_spo = spos[pred_index]
                for spo in spos:
                    if spo['predicate'] == '票房区域' and spo['subject'] == spos[pred_index]['object']['@value']:
                        complex_spo = {"object_type": {"inArea": "地点", "@value": "Number"},
                                       "predicate": '票房',
                                       "object": {'inArea': spo['object']['@value'],
                                                  '@value': spos[pred_index]['object']['@value']},
                                       "subject_type": "影视作品",
                                       "subject": spos[pred_index]['subject']}
                        break
                complex_spos.append(complex_spo)
                del (spos[pred_index])
            except ValueError:
                break
        return complex_spos

    def win_a_prize(spos):
        # 先合并 娱乐人物-》作品-》奖项
        works_spos = []
        while True:
            try:
                predicate_list = [spo['predicate'] for spo in spos]
                pred_index = predicate_list.index('作品')
                for spo in spos:  # 可能获得多个不同的奖
                    # 假如正确必定会有一次进入
                    if spo['predicate'] == '作品获奖' and spo['subject'] == spos[pred_index]['object']['@value']:
                        work_spo = {"object_type": {"inWork": "作品", "@value": "奖项"},
                                    "predicate": '人物-作品-奖项',
                                    "object": {'inWork': spo['subject'],
                                               '@value': spo['object']['@value']},
                                    "subject_type": "娱乐人物",
                                    "subject": spos[pred_index]['subject']}
                        works_spos.append(work_spo)
                del (spos[pred_index])
            except ValueError:
                break

        spos.extend(works_spos)
        complex_spos = []
        while True:
            try:
                predicate_list = [spo['predicate'] for spo in spos]
                pred_index = predicate_list.index('获奖')
                target_spo = spos[pred_index]
                complex_spo = {'object_type': {'@value': '奖项'},
                               'predicate': '获奖',
                               'object': {'@value': target_spo['object']['@value']},
                               'subject_type': '娱乐人物',
                               'subject': target_spo['subject']}
                flag1 = flag3 = True
                for spo in spos[:]:
                    if flag1 and spo['predicate'] == '奖项时间' and spo['subject'] == target_spo['object']['@value']:
                        complex_spo['object_type']['onDate'] = 'Date'
                        complex_spo['object']['onDate'] = spo['object']['@value']
                        del (spos[spos.index(spo)])
                        flag1 = False

                    elif flag3 and spo['predicate'] == '人物-作品-奖项' and spo['subject'] == target_spo['subject'] and \
                            spo['object']['@value'] == target_spo['object']['@value']:
                        complex_spo['object_type']['inWork'] = '作品'
                        complex_spo['object']['inWork'] = spo['object']['inWork']
                        del (spos[spos.index(spo)])
                        flag3 = False

                complex_spos.append(complex_spo)
                del (spos[spos.index(target_spo)])
            except ValueError:
                break
        return complex_spos

    def portray(spos):
        # 娱乐人物-》影视作品-》人物
        people_spos = []
        while True:
            try:
                predicate_list = [spo['predicate'] for spo in spos]
                pred_index = predicate_list.index('参与影视作品')
                for spo in spos:
                    if spo['predicate'] == '角色' and spo['subject'] == spos[pred_index]['object']['@value']:
                        complex_spo = {"object_type": {"inWork": "影视作品", "@value": "人物"},
                                       "predicate": '娱乐人物-影视作品-人物',
                                       "object": {'inWork': spo['subject'], '@value': spo['object']['@value']},
                                       "subject_type": "娱乐人物",
                                       "subject": spos[pred_index]['subject']}
                        people_spos.append(complex_spo)
                del (spos[pred_index])
            except ValueError:
                break

        complex_spos = []
        spos.extend(people_spos)
        while True:
            try:
                predicate_list = [spo['predicate'] for spo in spos]
                pred_index = predicate_list.index('饰演')
                # complex_spo = spos[pred_index]
                for spo in spos:
                    if spo['predicate'] == '娱乐人物-影视作品-人物' and spo['subject'] == spos[pred_index]['subject'] and \
                            spo['object']['@value'] == spos[pred_index]['object']['@value']:
                        complex_spo = {"object_type": {"inWork": "影视作品", "@value": "人物"},
                                       "predicate": '饰演',
                                       "object": {'inWork': spo['object']['inWork'], '@value': spo['object']['@value']},
                                       "subject_type": "娱乐人物",
                                       "subject": spos[pred_index]['subject']}
                        complex_spos.append(complex_spo)
                del (spos[pred_index])
            except ValueError:
                break
        return complex_spos

    complex_scheams = {'配音', '角色', '上映时间', '上映地点-时间', '票房', '票房区域',
                       '获奖', '作品', '作品获奖', '奖项时间', '饰演', '参与影视作品'}

    new_spos = []
    spos_waiting = []
    for spo in spos_inputs[:]:
        if spo['predicate'] in complex_scheams:
            spos_waiting.append(spo)
        else:
            new_spos.append(spo)
    new_spos.extend(dub(spos_waiting))
    new_spos.extend(release_date(spos_waiting))
    new_spos.extend(box_office(spos_waiting))
    new_spos.extend(win_a_prize(spos_waiting))
    new_spos.extend(portray(spos_waiting))
    return new_spos


def get_char2word_map(chars, words):
    char_index = 0
    word_index = 0
    char2word_map = [0] * len(chars)
    while True:
        if chars[char_index] in words[word_index]:
            char2word_map[char_index] = word_index
            char_index += 1
        elif word_index + 1 == len(words):
            break
        elif chars[char_index] in words[word_index + 1]:
            word_index += 1
            char2word_map[char_index] = word_index
            char_index += 1
        else:
            char2word_map[char_index] = word_index  # 暂时归结到前面一个去
            char_index += 1
        if char_index == len(chars):
            break
    return char2word_map  # 此方案： cut text的最后一个一定要加一个pad


def cut_tokenizer(words, word_dict):
    tokens = []
    for w in words:
        if w in word_dict.keys():
            tokens.append(word_dict[w])
        else:
            tokens.append(0)
    tokens.append(0)
    return tokens


def get_tokens(text, seg, tokenizer, word_dict):
    split_chars = ('，', ' ', '《', '》', '、', '：', '（', '）', '“', '”', '\xa0', '-', '.', '·', '\u3000', ':', '—', ',',
                   '/', '(', ')',
                   '…', '；', '【', '】', '~', '=', '#', '[', ']', '「', '」', '>', '%', '－', '"', '&', '!', '～', '@', '★',
                   '_', '*', '+',
                   '<', '?', '’', '‘', '．', '━', '|', '＞', '`', '☆', ';', '●', '―', "'", '『', '』', '═', '／', '•', '。',
                   '°', '・', '〉',
                   '〈', '◆', '\\', '▼', '→', '^', '℃', '▲', '┈', '×', '↓', '①', '◎', '﹏', '＜', '〔', '〕', '○', '１', '②',
                   'Ⅱ', '′', '◇',
                   '\ue5e5', '∶', '─', '–', '┅', '♥', '！', '$', '〓', '■', '┄', '←', '｜', '③', '２', '０')
    bert_tokens = tokenizer.encode(text, max_length=LIC2020Env.MAX_LENGTH)
    words = seg.cut(text)

    # get bert to word map
    char2word_map = get_char2word_map(text, words)
    bert2char_map = get_index_map(text, tokenizer)
    bert2word_map = [0] * len(bert2char_map)
    for bert_index in range(1, len(bert2char_map) - 1):
        bert2word_map[bert_index] = char2word_map[bert2char_map[bert_index]]
    bert2word_map[0] = -1
    bert2word_map[-1] = -1

    word_tokens = cut_tokenizer(words, word_dict)
    bert2word_seq = []
    split_feature = [0] * len(bert_tokens)
    for bert_index in range(len(bert_tokens)):
        bert2word_seq.append(word_tokens[bert2word_map[bert_index]])

        # handfeatures
        if 0 < bert_index < len(bert_tokens) - 1 and text[bert2char_map[bert_index]] in split_chars:
            split_feature[bert_index] = 1
    return bert_tokens, bert2word_seq, split_feature


def change_data_format(file_name, output_name):
    def simplify_schema(spos):
        new_spos = []
        for spo in spos:
            if spo['predicate'] == '丈夫':
                spo = {'subject_type': spo['object_type']['@value'],
                       'subject': spo['object']['@value'],
                       'predicate': '妻子',
                       'object_type': {'@value': spo['subject_type']},
                       'object': {'@value': spo['subject']}}
            if spo not in new_spos:
                new_spos.append(spo)
        return new_spos


    schema = LIC2020Schema()
    tokenizer = MyBertTokenizer.from_pretrained(BERT_MODEL)
    seg = pkuseg.pkuseg()
    word_dict = pickle.load(open(ROOT_DATA + 'w2v_vocab.pkl', 'rb'))
    file_name1 = ROOT_DATA + 'lic_2020/' + file_name
    output_name1 = ROOT_DATA + 'lic_2020/' + output_name
    output_name2 = ROOT_DATA + 'lic_2020/' + 'clean_' + file_name
    output_name3 = ROOT_DATA + 'lic_2020/' + 'decomposed_' + file_name

    new_data = []
    clean_data = []
    decomposed_data = []
    with open(file_name1, mode='r', encoding='utf-8') as fp:
        raw_data = fp.readlines()
        for iii, d in enumerate(raw_data):
            d = json.loads(d)

            # clean data
            new_d = {'text': d['text'], 'spo_list': []}
            d['spo_list'] = postprocess_1(d['text'], d['spo_list'])
            for spo in d['spo_list']:
                # 某些是没有值的
                if spo['subject']:
                    _object = {}
                    object_type = {}
                    for key in spo['object'].keys():  # 有些用例 object 有 period 但是object_type没有
                        # 某些是没有值的
                        if spo['object'][key]:
                            object_type[key] = schema.schema_dict[spo['predicate']]['object_type'][key]
                            _object[key] = spo['object'][key]

                        # 此类异常通常是{'subject_type': '图书作品', 'subject': '金毓黻与《中国史学史', 'predicate': '作者',
                        # 'object_type': {'@value': '人物'}, 'object': {'@value': '金毓黻与《中国史学史'}}
                        if spo['predicate'] not in ('改编自', '所属专辑', '主题曲') and \
                                spo['subject'] == spo['object'][key]:
                            _object = {}
                            break

                    if _object:
                        new_spo = {'subject_type': schema.schema_dict[spo['predicate']]['subject_type'],
                                   'subject': spo['subject'],
                                   'predicate': spo['predicate'],
                                   'object_type': object_type,
                                   'object': _object}
                        if new_spo not in new_d['spo_list']:
                            new_d['spo_list'].append(new_spo)

            if new_d['spo_list']:
                # simplify shcema
                simply_spos = simplify_schema(new_d['spo_list'])

                # expand spos
                e_spos = []

                for spo in simply_spos:
                    e_spos.extend(expand_spo(spo))

                # tokenizer
                bert_token, bert2word_seq, hand_features = get_tokens(new_d['text'], seg, tokenizer, word_dict)
                tokenizer_out = {'text': bert_token,
                                 'b2w': bert2word_seq,
                                 'hand': hand_features,
                                 'spo_list': []}
                expand_out = {'text': new_d['text'], 'spo_list': e_spos}
                for spo in e_spos:
                    objects = {}
                    for key in spo['object'].keys():
                        objects[key] = [schema.obj2id_dict[spo['object_type'][key]],
                                        tokenizer.encode(spo['object'][key], add_special_tokens=False)]
                    tokenizer_out['spo_list'].append({'subject': [schema.sbj2id_dict[spo['subject_type']],
                                                                  tokenizer.encode(spo['subject'],
                                                                                   add_special_tokens=False)],
                                                      'predicate': schema.pred2id_dict[spo['predicate']],
                                                      'object': objects})

                clean_data.append(new_d)
                decomposed_data.append(expand_out)
                new_data.append(tokenizer_out)

    def dump_file(dump_data, outfile):
        output_fp = open(outfile, 'w+', encoding='utf-8')
        output_fp.write('[')
        for ii, d in enumerate(dump_data):
            string = '\n ' + json.dumps(d, ensure_ascii=False)
            if ii < len(new_data) - 1:
                string += ','
            output_fp.write(string)
        output_fp.write('\n]')

    # dump file
    dump_file(new_data, output_name1)
    dump_file(clean_data, output_name2)
    dump_file(decomposed_data, output_name3)


def change_data_format_test(file_name, output_name):
    file_name = ROOT_DATA + 'lic_2020/' + file_name
    output_name = ROOT_DATA + 'lic_2020/' + output_name

    tokenizer = MyBertTokenizer.from_pretrained(BERT_MODEL)
    seg = pkuseg.pkuseg()
    word_dict = pickle.load(open(ROOT_DATA + 'w2v_vocab.pkl', 'rb'))

    output_fp = open(output_name, 'w+', encoding='utf-8')
    string = ''
    string += '['
    with open(file_name, mode='r', encoding='utf-8') as fp:
        raw_data = fp.readlines()
        for ii, d in enumerate(raw_data):
            d = json.loads(d)
            bert_token, bert2word_seq, hand_features = get_tokens(d['text'], seg, tokenizer, word_dict)
            d = {'text': bert_token,
                 'b2w': bert2word_seq,
                 'hand': hand_features}
            string += '\n ' + json.dumps(d, ensure_ascii=False)
            if ii < len(raw_data) - 1:
                string += ','
    string += '\n]'
    output_fp.write(string)
    output_fp.close()


def test_expand_and_combine():
    data = open(ROOT_DATA + 'lic_2020/clean_train_data.json', 'r', encoding='utf-8')
    data = json.load(data)
    correct_num = 0
    pred_num = 0
    true_num = 0
    for d in data:
        e_spos = []
        for spo in d['spo_list']:
            e_spos.extend(expand_spo(spo))
        c_spos = combine_spos(e_spos)
        for spo in c_spos:
            if spo in d['spo_list']:
                correct_num += 1
        pred_num += len(c_spos)
        true_num += len(d['spo_list'])
    print(calculate_f1(correct_num, pred_num, true_num, verbose=True))


def filter(in_file, out_file):
    file_pred = open(in_file, 'r', encoding='utf-8')
    out_file = open(out_file, 'w+', encoding='utf-8')
    data = file_pred.readlines()
    for d in data:
        d = json.loads(d)
        d['spo_list'] = postprocess(d['text'], d['spo_list'])
        d['spo_list'] = combine_spos(d['spo_list'])
        d['spo_list'] = postprocess_1(d['text'], d['spo_list'])
        string = json.dumps(d, ensure_ascii=False) + '\n'
        out_file.write(string)
    out_file.close()


from multiprocessing import Process


def info(filename):
    fp = open(filename, 'r', encoding='utf-8')
    raw_data = fp.readlines()
    data = []
    for d in raw_data:
        data.append(json.loads(d))
    sign_dict = {}

    # data = json.load(fp)

    def find_blank(entity):
        for s in entity:
            if s == ' ':
                print('bbbbb:', entity)
                return True
            elif s == '\xa0':
                print('a0   :', entity)
                return True
            elif s == '\u3000':
                print('u3000:', entity)
                return True
        return False

    def check_company(text, entity_type, entity):
        if entity_type in ('企业', '机构', '企业/品牌'):
            # print(entity)
            # return
            flag = True
            while flag:
                flag = False
                points = locate_entity(text, entity, [[0, 0]])
                for s, e in points:
                    # if len(text[e+1:e + 3]) == 2 and text[e+1:e + 3] in ('管理'):
                    if text[e + 1:e + 3] in ('公司', '有限', '股份', '责任', '集团', '科技', '集团', '管理', '汽车',):
                        print(entity, '->', text[s:e + 3])
                        print(text)
                        entity = text[s:e + 3]
                        flag = True
                        break

    for d in data:
        #     for s in d['text']:
        #         if s.isalpha() or is_chinese(s) or s in '0123456789':
        #             continue
        #         if s in sign_dict.keys():
        #             sign_dict[s] += 1
        #         else:
        #             sign_dict[s] = 1
        # a = [i for i in sign_dict.items()]
        # a.sort(key=lambda x: x[1], reverse=True)
        # a = [i[0] for i in a if i[1] > 50]
        # print(a)
        for spo in d['spo_list']:
            check_company(d['text'], spo['subject_type'], spo['subject'])
            for key in spo['object'].keys():
                check_company(d['text'], spo['object_type'][key], spo['object'][key])
                # find_blank(spo['object'][key])


def compare_two_file(file_0, file_1):
    fp = open(file_0, 'r', encoding='utf-8')
    raw_data = fp.readlines()
    data0 = []
    for d in raw_data:
        data0.append(json.loads(d))

    fp = open(file_1, 'r', encoding='utf-8')
    raw_data = fp.readlines()
    data1 = []
    for d in raw_data:
        data1.append(json.loads(d))

    for d0, d1 in zip(data0, data1):
        for spo_0 in d0['spo_list']:
            if spo_0 not in d1['spo_list']:
                print('000:', spo_0)
        for spo_1 in d1['spo_list']:
            if spo_1 not in d0['spo_list']:
                print('111:', spo_1)


def get_correct_result(files, out_file):
    data_list = []
    cc = 0
    all = 0
    out_fp = open(out_file, 'w+', encoding='utf-8')
    out_1_fp = open(ROOT_RESULT+'temp_111.json', 'w+', encoding='utf-8')
    for file in files:
        data = []
        fp = open(file, 'r', encoding='utf-8')
        raw_data = fp.readlines()
        for line in raw_data:
            data.append(json.loads(line))
        fp.close()
        data_list.append(data)
    for lines in zip(*data_list):
        new_d = {'text': lines[0]['text'], 'spo_list': []}
        new_d_1 = {'text': lines[0]['text'], 'spo_list': []}
        temp_spos = []
        for line in lines:
            for spo in line['spo_list']:
                if spo not in temp_spos:
                    all += 1
                    temp_spos.append(spo)
        for spo in temp_spos:
            if std_kg.check_spo_1(spo):
                cc += 1
                new_d['spo_list'].append(spo)
                continue

            count = 0
            for line in lines:
                if spo in line['spo_list']:
                    count += 1
            if count >= 3:
                cc += 1
                new_d['spo_list'].append(spo)
                continue
            new_d_1['spo_list'].append(spo)
        string = json.dumps(new_d, ensure_ascii=False) + '\n'
        out_fp.write(string)
        string = json.dumps(new_d_1, ensure_ascii=False) + '\n'
        out_1_fp.write(string)
    print(all, cc)
    out_fp.close()
    out_1_fp.close()

    # for file in temp_files:
    #     os.remove(file)


def combine_results(files, main_file, out_file):
    get_correct_result(files, ROOT_RESULT + 'temp_0.json')
    c_files = [main_file, ROOT_RESULT + 'temp_0.json']

    raw_data_list = []
    temp_out = open(ROOT_RESULT+'temp_out.json', 'w+', encoding='utf-8')
    for file in c_files:
        raw_data_list.append(open(file, 'r', encoding='utf-8').readlines())

    for i, lines in enumerate(zip(*raw_data_list)):
        lines = [json.loads(line) for line in lines]
        new_d = {'text': lines[0]['text'], 'spo_list': []}
        for d in lines:
            for spo in d['spo_list']:
                if spo not in new_d['spo_list']:
                    new_d['spo_list'].append(spo)
        string = json.dumps(new_d, ensure_ascii=False) + '\n'
        temp_out.write(string)
    temp_out.close()
    filter(ROOT_RESULT+'temp_out.json', out_file)
    # os.remove(ROOT_RESULT+'temp_0.json')
    # os.remove(ROOT_RESULT+'temp_out.json')


# ===============================================================
class LIC2020Env(object):
    Data = LIC2020Data
    Schema = LIC2020Schema
    NUM_SCHEMA = 54
    NUM_SBJ_TYPE = 20
    MAX_LENGTH = 205
    generate_formal_results = generate_formal_results
    get_formal_result = get_formal_result
    combine_results = combine_results


if __name__ == '__main__':
    print("LIC 2020 DATA PREPARE")
    # p1 = Process(target=change_data_format, args=('train_data.json', 'train_tokenizer.json'))
    # p1.start()
    # p2 = Process(target=change_data_format, args=('dev_data.json', 'dev_tokenizer.json'))
    # p2.start()
    # p3 = Process(target=change_data_format_test, args=('test2_data.json', 'test2_tokenizer.json'))
    # p3.start()
    # p1.join()
    # p2.join()
    # p3.join()

    # combine_results([ROOT_RESULT + '8012_correct.json',
    #                  ROOT_RESULT + '8017_correct.json',
    #                  ROOT_RESULT + 'ave3_test_joint.json',
    #                  ROOT_RESULT + '7993_correct.json'],
    #                 ROOT_RESULT + 'combine_result.json')
    # filter(ROOT_RESULT + 'temp_0.json', ROOT_RESULT + 'filter_temp_0.json')
    # filter(ROOT_RESULT + 'temp_111.json', ROOT_RESULT + 'filter_temp_111.json')
    # filter(ROOT_RESULT + 'test1_joint.json', ROOT_RESULT + 'filter_test1_joint.json')
    # compare_two_file(ROOT_RESULT + 'combine_result_base.json', ROOT_RESULT + 'combine_result_1.json')
#
