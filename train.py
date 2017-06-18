import sys
import getopt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Masking
from keras.callbacks import ModelCheckpoint

from yaoai import setup


def main(argv):
    sentence_length = 40
    dataX, dataY, story, word_to_int, int_to_word = setup(sentence_length, './input.txt')

    print(dataX)

    model = Sequential([
        Masking(mask_value=0.0, input_shape=(dataX.shape[1:])),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        Dense(dataY.shape[2], activation='softmax'),
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    file_path = 'word-LSTM-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(dataX, dataY, epochs=80, batch_size=sentence_length, callbacks=callbacks_list)

    file_path = 'model.hdf5'
    model.save(file_path)

if __name__ == '__main__':
    main(sys.argv)
