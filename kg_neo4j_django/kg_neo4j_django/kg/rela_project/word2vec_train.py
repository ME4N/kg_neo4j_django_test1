import re
from gensim import models
import os

data_path = '../data/关系抽取222.txt'


def load_data():
    ff = open(data_path)
    ff_out = open('./data_handle/sample.txt', 'w', encoding='utf-8')
    ff_out_rela = open('./data_handle/relation2id.txt', 'w', encoding='utf-8')
    ff_w2v_train = open('./data_handle/w2v_train_sample.txt', 'w', encoding='utf-8')
    rela_map = {}
    sentence_all = []
    for line in ff.readlines():
        sentence = []
        rela = line.strip('\n').split(" ")[0].replace("\ufeff", "")
        e1 = re.findall(".*<e1>(.*)</e1>.*", line)[0]
        e2 = re.findall(".*<e2>(.*)</e2>.*", line)[0]
        line_w = line.strip('\n').split(" ")[1].replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace(
            "</e2>", "")
        out_str = '\t'.join([e1, e2, rela, line_w])
        ff_out.write(out_str + '\n')
        if rela not in rela_map:
            rela_map[rela] = len(rela_map) + 1
        for w in line_w:
            if w not in ['，', '。', ',', '！', '、']:
                sentence.append(w)
        sentence_all.append(sentence)
        ff_w2v_train.write(' '.join(sentence) + '\n')
    ff_out_rela.write('unknown' + '\t' + '0' + '\n')
    for k in rela_map:
        ff_out_rela.write(k + '\t' + str(rela_map[k]) + '\n')

    return sentence_all


def word2vec_train(sentences):
    vec_dim = 64
    min_word_count = 5
    window = 2
    sample = 1e-5
    negative = 5
    sg = 1
    hs = 1
    iter = 5

    model_path = './w2v_model/word2vec.model'
    model = models.Word2Vec(sentences, iter=iter, size=vec_dim, window=window,
                            min_count=min_word_count, negative=negative, sample=sample, sg=sg, hs=hs)

    model.save(os.path.join('./', model_path))
    model.wv.save_word2vec_format(os.path.join('./data_handle/', 'vec.txt'), binary=False)


def train_test_split():
    ff = open('./data_handle/sample.txt')
    ff_train = open('./data_handle/train.txt', 'w', encoding='utf-8')
    ff_test = open('./data_handle/test.txt', 'w', encoding='utf-8')
    sum_all = len(ff.readlines())
    ff = open('./data_handle/sample.txt')
    i = 0
    for line in ff.readlines():
        if i < sum_all / 10 * 9:
            ff_train.write(line)
        else:
            ff_test.write(line)
        i += 1


if __name__ == '__main__':
    sentence_all = load_data()
    word2vec_train(sentence_all)
    train_test_split()
