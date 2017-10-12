'''This is based on the blog post by fchollet, the author of keras, see
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
'''


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop

from utils import data_path, download_names, shuffle_names

import numpy as np
import os.path
import random
import sys


batch_size = 128
maxlen = 15
step = 3


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def predict(model, text, maxlen, len_chars, indices_char, diversities, N=400):
    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in diversities:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(N):
            x = np.zeros((1, maxlen, len_chars))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()


if __name__ == '__main__':
    if not os.path.isfile(data_path):
        print('download data from Wikipedia')
        download_names()
        shuffle_names()

    text = open(data_path).read()
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    for iteration in range(1, 10):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=batch_size, epochs=1)
        diversities = [0.25, 0.5, 0.75]
        predict(model, text, maxlen, len(chars), indices_char, diversities)

    fin_div = [0.75]
    predict(model, text, maxlen, len(chars), indices_char, fin_div, N=20000)


