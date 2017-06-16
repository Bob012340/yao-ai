import re
import sys
import yaoai

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Masking


sentence_length = 40
dataX, dataY, story, word_to_int, int_to_word = yaoai.setup(sentence_length)

model = Sequential([
    Masking(mask_value=0.0, input_shape=(dataX.shape[1:])),
    LSTM(256, return_sequences=True),
    Dropout(0.2),
    LSTM(256, return_sequences=True),
    Dropout(0.2),
    Dense(dataY.shape[2], activation='softmax'),
])
model.load_weights('./model.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='adam')

model_output = model.predict(dataX)
file_path = './output.txt'
file = open(file_path, 'w')
for x in model_output:
    output_data = []
    for y in x:
        output_data.append(int_to_word[np.argmax(y)])
    output_data.append('\n\n')
    file.writelines(output_data)
file.close()
