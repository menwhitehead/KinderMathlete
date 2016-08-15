import random
import numpy as np
# np.random.seed(42)  # for reproducibility

from keras.preprocessing import sequence
# from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, TimeDistributed, Masking

from keras.layers import LSTM, SimpleRNN, GRU

max_features = 13000
maxlen = 50  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

calc_buttons = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '.', ' ']

def to_one_hot(x):
    # result = np.zeros(len(calc_buttons))
    result = [0.0] * len(calc_buttons)
    result[calc_buttons.index(x)] = 1
    return result


print('Loading data...')
X_train = []
y_train = []

X_test_orig = []
y_test_orig = []

train_prob = 0.9

f = open("seq2seq.txt", 'r')
#all_txt = f.read()
#f.close()
words = {}

for line in f:
    tokens = line.split()
    for token in tokens[:-1]:
        words[token] = 1

uniques = words.keys()
uniques.sort()
# print uniques

vocab_size = len(uniques)

print "VOCAB SIZE:", vocab_size
PAD_LENGTH = 10
max_features = vocab_size


f = open("seq2seq.txt", 'r')
for line in f:
    tokens = line.split()
    answer = tokens[-1]
    seq = []
    for token in tokens[:-1]:
        txt = uniques.index(token)
        seq.append(txt)

    ans_seq = []
    for char in answer:
        #ans_seq.append(to_one_hot(char))
        #ans_seq.append(to_one_hot(char))
        ans_seq.append(calc_buttons.index(char))
    # while len(ans_seq) < PAD_LENGTH:
    #     ans_seq.append(to_one_hot(' '))
    # ans_seq = np.array(ans_seq)
    #ans = uniques.index(answer)
    #print(seq)
    #print(ans)
    if random.random() < train_prob:
        X_train.append(seq)
        y_train.append(ans_seq)
    else:
        X_test_orig.append(seq)
        y_test_orig.append(ans_seq)

    #print y_train

print "train sequences:", len(X_train)
print "test sequences:", len(X_test_orig)

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test_orig, maxlen=maxlen)

y_train = sequence.pad_sequences(y_train, padding="post", maxlen=PAD_LENGTH)
y_test = sequence.pad_sequences(y_test_orig, padding="post", maxlen=PAD_LENGTH)

print X_train.shape, X_test.shape, y_train.shape, y_test.shape
print X_train[0], y_train[0]


# X_train = np.reshape(X_train, X_train.shape + (1,))
# X_test = np.reshape(X_test, X_test.shape + (1,))
# y_train = np.reshape(y_train, y_train.shape + (1,))
# y_test = np.reshape(y_test, y_test.shape + (1,))

number_hidden = 256
number_output = len(calc_buttons)
length_output = 10

import seq2seq
from seq2seq.models import SimpleSeq2seq

model = SimpleSeq2seq(input_dim=(maxlen, 1), hidden_dim=number_hidden, output_length=length_output, output_dim=number_output)
model.compile(loss='mse', optimizer='rmsprop')

print('Train...')

# predictions = model.predict(X_test)
# print predictions

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=50, validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Score:', score)
print('Accuracy:', acc)

predictions = model.predict(X_test)
for i in range(len(predictions)):
    p = predictions[i]
    ind = list(p).index(max(p))
    token = uniques[ind]
    answer = uniques[y_test_orig[i]]
    for word in range(len(X_test_orig[i])):
        print uniques[X_test_orig[i][word]],
    print token, answer
