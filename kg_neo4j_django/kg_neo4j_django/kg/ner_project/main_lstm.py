import tensorflow as tf

import numpy as np
import os, argparse, time
from .model import BiLSTM_CRF
from .utils import str2bool, get_logger, get_entity
from .data import read_corpus, read_dictionary, tag2label_mapping, random_embedding, vocab_build, \
    build_character_embeddings

# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # need ~700MB GPU memory

path_org = os.path.split(os.path.realpath(__file__))[0]

class Args(object):
    def __init__(self):
        self.dataset_name = 'ner'
        self.batch_size=1
        self.epoch=10
        self.hidden_dim=32
        self.optimizer='Adam'
        self.CRF=True
        self.lr=0.01
        self.clip=5.0
        self.dropout=1.0
        self.update_embedding = True
        self.use_pre_emb =False
        self.pretrained_emb_path='sgns.wiki.char'
        self.embedding_dim =32
        self.shuffle = True
        self.mode = 'train' #train/test/demo


args = Args()

# vocabulary build
if not os.path.exists(os.path.join(path_org+'/../data', args.dataset_name, 'word2id.pkl')):
    vocab_build(os.path.join(path_org+'/../data', args.dataset_name, 'word2id.pkl'),
                os.path.join(path_org+'/../data', args.dataset_name, 'train_data.txt'))

# get word dictionary
word2id = read_dictionary(os.path.join(path_org+'/../data', args.dataset_name, 'word2id.pkl'))

# build char embeddings
if not args.use_pre_emb:
    embeddings = random_embedding(word2id, args.embedding_dim)
    log_pre = 'not_use_pretrained_embeddings'
else:
    pre_emb_path = os.path.join('.', args.pretrained_emb_path)
    embeddings_path = os.path.join('data_path', args.dataset_name, 'pretrain_embedding.npy')
    if not os.path.exists(embeddings_path):
        build_character_embeddings(pre_emb_path, embeddings_path, word2id, args.embedding_dim)
    embeddings = np.array(np.load(embeddings_path), dtype='float32')
    log_pre = 'use_pretrained_embeddings'

# choose tag2label
tag2label = tag2label_mapping[args.dataset_name]

# read corpus and get training data
if args.mode != 'demo':
    train_path = os.path.join(path_org+'/../data', args.dataset_name, 'train_data.txt')
    test_path = os.path.join(path_org+'/../data', args.dataset_name, 'test_data.txt')
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path)
    test_size = len(test_data)

# paths setting
paths = {}
output_path = os.path.join(path_org+'/model_path', args.dataset_name)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, args.dataset_name + log_pre + "_log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))

def train():
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    # train model on the whole training data
    print("train data: {}".format(len(train_data)))
    print("test data: {}".format(test_size))
    model.train(train=train_data, dev=test_data)  # use test_data.txt as the dev_data to see overfitting phenomena

def predict(sentence):
    tf.reset_default_graph()
    ckpt_file = tf.train.latest_checkpoint(model_path)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    print("--------------------------")
    if sentence=='':
        return ''

    model.build_graph()
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, ckpt_file)
        demo_sent = list(sentence.strip())
        demo_data = [(demo_sent, ['O'] * len(demo_sent))]
        tag = model.demo_one(sess, demo_data)
        print("tag:", tag)
        center_word = get_entity(tag, demo_sent)
        return center_word


if __name__ == '__main__':
    train()

