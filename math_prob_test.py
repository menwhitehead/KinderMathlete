from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb

from keras.utils.np_utils import to_categorical

max_features = 13000
maxlen = 50  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
X_train = []
y_train = []

f = open("cleaned.txt", 'r')
all_txt = f.read()
f.close()
tokens = all_txt.split()
words = {}
for token in tokens:
    words[token] = 1

uniques = words.keys()
uniques.sort()


f = open("cleaned.txt", 'r')
for line in f:
    tokens = line.split()
    answer = tokens[-1]
    seq = []
    for token in tokens[:-1]:
        txt = uniques.index(token)
        seq.append(txt)
    ans = uniques.index(answer)
    #print(seq)
    #print(ans)

    X_train.append(seq)
    y_train.append(ans)
    

print(len(X_train), 'train sequences')
y_train = to_categorical(y_train)

#print(X_train)
#print(y_train.shape)
print(y_train)

#print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
print('X_train shape:', X_train.shape)


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(len(y_train[0])))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
#model.compile(loss='binary_crossentropy',
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=150)
score, acc = model.evaluate(X_train, y_train, batch_size=batch_size)
print('Score:', score)
print('Accuracy:', acc)
