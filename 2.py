# coding: utf-8
import numpy as np
from rnnmath import *
from rnn1 import *
import pandas as pd
from rnnmath import *
from utils import *

vocab_size = 2000
data_folder = 'data'


    #     print xx
vocab = pd.read_table(data_folder + "/vocab.ptb.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
num_to_word = dict(enumerate(vocab.index[:vocab_size]))
word_to_num = invert_dict(num_to_word)

# calculate loss vocabulary words due to vocab_size
fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
print("Retained %d words from %d (%.02f%% of all tokens)\n" % (vocab_size, len(vocab), 100*(1-fraction_lost)))

docs = load_dataset(data_folder + '/ptb-train.txt')
S_train = docs_to_indices(docs, word_to_num)
X_train, D_train = seqs_to_lmXY(S_train)

print len(X_train),len(D_train)