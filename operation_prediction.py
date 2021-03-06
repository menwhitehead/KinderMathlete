import random
import numpy as np
np.random.seed(42)  # for reproducibility

from keras.preprocessing import sequence
# from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU

max_features = 13000
maxlen = 50  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
X_train = []
y_train = []

X_test = []
y_test = []

train_prob = 0.9

f = open("operation_prediction.txt", 'r')
all_txt = f.read()
f.close()
tokens = all_txt.split()
words = {}
for token in tokens:
    words[token] = 1

uniques = words.keys()
uniques.sort()
vocab_size = len(uniques)

print "VOCAB SIZE:", vocab_size

f = open("operation_prediction.txt", 'r')
for line in f:
    tokens = line.split()
    answer = tokens[-1]
    seq = []
    for token in tokens[:-1]:
        txt = uniques.index(token)
        seq.append(txt)
    ans = uniques.index(answer)

    if random.random() < train_prob:
        X_train.append(seq)
        y_train.append(ans)
    else:
        X_test.append(seq)
        y_test.append(ans)


print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
y_train.append(vocab_size+1)
y_test.append(vocab_size+1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train = y_train[:-1]
y_test = y_test[:-1]
#print(X_train)
#print(y_train.shape)

#print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)


print X_train.shape, X_test.shape, y_train.shape, y_test.shape
print X_train[0], y_train[0]

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.2))
# model.add(LSTM(256, dropout_W=0.2, dropout_U=0.2))
model.add(LSTM(256, dropout_W=0.5, dropout_U=0.5))
model.add(Dense(len(y_train[0])))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=100, validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Score:', score)
print('Accuracy:', acc)
