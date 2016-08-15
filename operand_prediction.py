import random
import numpy as np
# np.random.seed(42)  # for reproducibility

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

X_test_orig = []
y_test_orig = []

train_prob = 0.9

f = open("operand_prediction.txt", 'r')
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


f = open("operand_prediction.txt", 'r')
for line in f:
    tokens = line.split()
    answer = tokens[-2]
    seq = []
    for token in tokens[:-2]:
        txt = uniques.index(token)
        seq.append(txt)

    ans = uniques.index(answer)
    #print(seq)
    #print(ans)
    if random.random() < train_prob:
        X_train.append(seq)
        y_train.append(ans)
    else:
        X_test_orig.append(seq)
        y_test_orig.append(ans)

print "train sequences:", len(X_train)
print "test sequences:", len(X_test_orig)
y_train.append(vocab_size+1)
y_test_orig.append(vocab_size+1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test_orig)
y_train = y_train[:-1]
y_test = y_test[:-1]
#print(X_train)
#print(y_train.shape)

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test_orig, maxlen=maxlen)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.2))
# model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
model.add(LSTM(256, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(len(y_train[0])))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
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
