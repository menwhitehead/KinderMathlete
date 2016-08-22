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
biggest_number = 10000

embedding_size = 128
lstm_size = 512
output_size = 204
dropout = 0.2

number_trials = 10000
epochs_per_trial = 10

def convertToOneHot(val, size):
    x = np.zeros(size)
    x[val] = 1.0
    return x

def buildModel():
    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen, dropout=dropout))
    # model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
    model.add(LSTM(lstm_size, dropout_W=dropout, dropout_U=dropout))
    model.add(Dense(output_size))
    # model.add(Activation('softmax'))
    model.add(Activation('sigmoid'))

    model.compile(loss='mse',
                  optimizer='adam')
                #   metrics=['accuracy'])
    return model

def loadVocab(filename="3answer_easy.txt"):
    f = open(filename, 'r')
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

    return uniques


def loadDataset(vocab, filename="3answer_easy.txt"):
    print('Loading data...')
    X_train_orig = []
    y_train = []

    X_test_orig = []
    y_test = []

    train_prob = 0.9

    f = open(filename, 'r')
    for line in f:
        tokens = line.split()
        operator, operand1, operand2 = tokens[-3:]

        seq = []
        for token in tokens[:-3]:
            txt = vocab.index(token)
            seq.append(txt)

        ans1 = vocab.index(operator)
        ans2 = vocab.index(operand1)
        ans3 = vocab.index(operand2)

        if operator == "+":
            operator_vector = convertToOneHot(0, 4)
        elif operator == "-":
            operator_vector = convertToOneHot(1, 4)
        elif operator == "*":
            operator_vector = convertToOneHot(2, 4)
        elif operator == "/":
            operator_vector = convertToOneHot(3, 4)

        operand_vector1 = convertToOneHot(int(operand1), 100)
        operand_vector2 = convertToOneHot(int(operand2), 100)

        combined = np.concatenate((operator_vector, operand_vector1, operand_vector2))
        # print combined

        if random.random() < train_prob:
            X_train_orig.append(seq)
            y_train.append(combined)
        else:
            X_test_orig.append(seq)
            y_test.append(combined)

    print "train sequences:", len(X_train_orig)
    print "test sequences:", len(X_test_orig)

    X_train = sequence.pad_sequences(X_train_orig, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test_orig, maxlen=maxlen)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test, X_train_orig, X_test_orig

def accuracyTest(model, X_test, y_test, X_test_orig):
    predictions = model.predict(X_test)
    correct = 0
    for i in range(len(predictions)):
        p = predictions[i]
        operator_vec, operand_vec1, operand_vec2 = np.split(p, [4, 104])
        operator_ans, operand_ans1, operand_ans2 = np.split(y_test[i], [4, 104])

        op_ind = list(operator_vec).index(max(operator_vec))
        op1_ind = list(operand_vec1).index(max(operand_vec1))
        op2_ind = list(operand_vec2).index(max(operand_vec2))

        ans_ind = list(operator_ans).index(max(operator_ans))
        ans1_ind = list(operand_ans1).index(max(operand_ans1))
        ans2_ind = list(operand_ans2).index(max(operand_ans2))

        # for word in range(len(X_test_orig[i])):
        #     print vocab[X_test_orig[i][word]],
        # print
        # print op_ind, op1_ind, op2_ind
        # print ans_ind, ans1_ind, ans2_ind

        if ans_ind == op_ind and ans1_ind == op1_ind and ans2_ind == op2_ind:
            correct += 1
    return correct * 100.0 / len(y_test)

vocab = loadVocab()
X_train, y_train, X_test, y_test, X_train_orig, X_test_orig = loadDataset(vocab)
model = buildModel()


print('Train...')
for trial in range(number_trials):
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs_per_trial, validation_data=(X_test, y_test), verbose=False)
    train_acc = accuracyTest(model, X_train, y_train, X_test_orig)
    test_acc = accuracyTest(model, X_test, y_test, X_test_orig)
    print "TRIAL %4d:     Train: %5.1f%%   Test: %5.1f%%" % (trial, train_acc, test_acc)
