from torch.utils.data import Dataset

from utils import MyBertTokenizer
import json as js

from global_config import ROOT_DATA, TEST_MODE


# max length 定为205 包括cls 和 seq


class RawDataSet(Dataset):
    def __init__(self, file_name):
        file_name = ROOT_DATA + 'lic_2019/' + file_name
        with open(file_name, mode='r', encoding='utf-8') as fp:
            self.data = js.load(fp)

        if TEST_MODE:
            self.len = 2000
        else:
            self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index]


class LIC2019Data(object):
    data_set = {"train": "train_tokenzier.json",
                "dev": "dev_tokenzier.json",
                "test": "dev_tokenzier.json"}

    def __new__(cls, mode):
        if mode in cls.data_set.keys():
            return RawDataSet(cls.data_set[mode])
        elif mode == 'schema':
            return LIC2019Schema()
        else:
            assert 0


class LIC2019Schema(object):
    def __init__(self):
        file_name = ROOT_DATA + 'lic_2019/schema.json'
        with open(file_name, mode='r', encoding='utf-8') as fp:
            data = fp.readlines()
            self.data = [js.loads(d) for d in data]
        self.predicate_list = list(set([d['predicate'] for d in self.data]))
        self.subjuect_type_list = list(set([d['subject_type'] for d in self.data]))
        self.object_type_list = list(set([d['object_type'] for d in self.data]))

        self.pred2id_dict = dict((pred, i) for i, pred in enumerate(self.predicate_list))
        self.id2pred_dict = dict((i, pred) for i, pred in enumerate(self.predicate_list))

        self.obj2id_dict = dict((obj, i) for i, obj in enumerate(self.object_type_list))
        self.id2obj_dict = dict((i, obj) for i, obj in enumerate(self.object_type_list))

        self.sbj2id_dict = dict((sbj, i) for i, sbj in enumerate(self.subjuect_type_list))
        self.id2sbj_dict = dict((i, sbj) for i, sbj in enumerate(self.subjuect_type_list))

    def pred2id(self, pred_list):
        return [self.pred2id_dict[pred] for pred in pred_list]

    def id2pred(self, id_list):
        return [self.id2pred_dict[idd] for idd in id_list]


class LIC2019Env(object):
    Data = LIC2019Data
    Schema = LIC2019Schema
    NUM_SCHEMA = 50
    MAX_LENGTH = 100


def change_data_format(file_name, output_name):
    file_name = ROOT_DATA + 'lic_2019/' + file_name
    output_name = ROOT_DATA + 'lic_2019/' + output_name
    with open(file_name, mode='r', encoding='utf-8') as fp:
        raw_data = fp.readlines()
        new_data = []
        for d in raw_data:
            d = js.loads(d)
            nd = {'text': d['text'], 'spo_list': []}
            for spo in d['spo_list']:
                subject = [spo['subject_type'], spo['subject']]
                predicate = spo['predicate']
                objects = {'@value': [spo['object_type'], spo['object']]}
                nd['spo_list'].append({'subject': subject, 'predicate': predicate, 'object': objects})
            if nd['spo_list']:
                new_data.append(nd)

        output_fp = open(output_name, 'w+', encoding='utf-8')
        js.dump(new_data, output_fp, ensure_ascii=False)
        output_fp.close()


def data_prepare(file_name, output_name):
    schema = LIC2019Schema()
    tokenzier = MyBertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    file_name = ROOT_DATA + 'lic_2019/' + file_name
    output_name = ROOT_DATA + 'lic_2019/' + output_name

    with open(file_name, mode='r', encoding='utf-8') as fp:
        raw_data = js.load(fp)
        new_data = []
        for d in raw_data:
            nd = {'text': tokenzier.encode(d['text'], max_length=LIC2019Env.MAX_LENGTH), 'spo_list': []}

            for spo in d['spo_list']:
                subject = (schema.sbj2id_dict[spo['subject'][0]],
                           tokenzier.encode(spo['subject'][1], add_special_tokens=False))
                predicate = schema.pred2id_dict[spo['predicate']]
                objects = {}
                for key in spo['object'].keys():
                    objects[key] = (schema.obj2id_dict[spo['object'][key][0]],
                                    tokenzier.encode(spo['object'][key][1], add_special_tokens=False))
                nd['spo_list'].append({'subject': subject, 'predicate': predicate, 'object': objects})
            if nd['spo_list']:
                new_data.append(nd)

        out_fp = open(output_name, 'w+', encoding='utf-8')
        js.dump(new_data, out_fp, ensure_ascii=False)
        out_fp.close()


if __name__ == '__main__':
    print("LIC 2019 DATA PREPARE")
    # change_data_format('train_data.json', 'train_rebuild.json')
    # change_data_format('dev_data.json', 'dev_rebuild.json')

    data_prepare('train_rebuild.json', 'train_tokenzier.json')
    data_prepare('dev_rebuild.json', 'dev_tokenzier.json')

