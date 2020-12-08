import pkuseg
from gensim.models import word2vec
import json
import pickle
from global_config import *
import numpy as np


def change_w2v_format():
    raw_file = ROOT_DATA + 'sgns.merge.word'
    vocab_file = ROOT_DATA + 'w2v_vocab.pkl'
    vec_file = ROOT_DATA + 'w2v_vector.pkl'
    raw_fp = open(raw_file, 'r', encoding='utf-8')
    vocab_fp = open(vocab_file, 'wb+')
    vec_fp = open(vec_file, 'wb+')
    raw_data = raw_fp.readlines()[1:]
    vocab_list = {}
    vocab_list['PAD'] = 0
    vec_list = [[0.0] * 300]
    for index, d in enumerate(raw_data):
        d = d.split()
        vocab_list[d[0]] = index + 1
        vec = [float(s) for s in d[1:]]
        vec_list.append(vec)
    print('ffff')
    pickle.dump(vocab_list, vocab_fp)
    pickle.dump(vec_list, vec_fp)

if __name__ == '__main__':
    # cut_files()
    # train_word2vec(ROOT_DATA + 'lic_2020/' + 'segment.txt')

    # en_wiki_word2vec_model = word2vec.Word2Vec.load(ROOT_DATA + 'lic_2020/w2v_model')
    #
    # testwords = ['金融', '上', '股票', '跌', '经济']
    # for i in range(5):
    #     res = en_wiki_word2vec_model.most_similar(testwords[i])
    #     print(testwords[i])
    #     print(res)
    change_w2v_format()
    # vocab_file = ROOT_DATA + 'w2v_vocab.pkl'
    # fp = open(vocab_file, 'rb')
    # a = pickle.load(fp)
    # print(11)

