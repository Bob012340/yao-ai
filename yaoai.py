import re

import numpy as np

from keras.utils import to_categorical


def read_input(filename):
    print('Opening and loading input file...')
    # Open file; read it; then separate words, punctuation & escape sequences into a list
    story_raw = re.findall(r"[\w']+|[.,!?;:\"\n]", open(filename, 'r').read().lower())

    # Generate a sorted dictionary
    int_to_word = sorted(set(story_raw))
    # int_to_word.append('\0')  # Add null character to it
    # int_to_word.sort()  # Sort again

    word_to_int = dict((c, i) for i, c in enumerate(int_to_word))  # Create a dictionary of the reverse of dictionary
    return story_raw, int_to_word, word_to_int


# Generate the story list of sentences
def generate_story(raw, sentence_length):
    print('Generating story list...')
    story, sentence = [], []
    for word in raw:
        sentence.append(word)
        if word in ['.', '!', '?', '.\"', '\n']:
            if len(sentence) > 1 and sentence[0][0] != sentence[0][0].lower:
                while len(sentence) < sentence_length:       # Each sentence is made to be 40 words & punctuation long
                    sentence.append(0)
                story.append(sentence)
            sentence = []
    return story


def generate_story2(raw, sentence_length):
    story, sentence = [], []
    for word in raw:
        sentence.append(word)
        if word in ['.', '!', '?', '.\"', '\n']:
            if len(sentence) > 1 and sentence[0][0] != sentence[0][0].lower:
                story.append(sentence)
            sentence = []
    return story


def encode(input_data, sentence_length, dictionary):
    print('Encoding story...')
    dataX = np.zeros((len(input_data), sentence_length-1, 1))
    dataY_init = np.zeros((len(input_data), sentence_length-1, 1))
    length_string = str(len(input_data))
    for sentence_index, sentence in enumerate(input_data):
        data = []
        for word_index, word in enumerate(sentence):
            if word != 0:
                data.append([(dictionary[word]/len(dictionary))])
            else:
                data.append([word])
        dataX[sentence_index] = data[:-1]
        dataY_init[sentence_index] = data[1:]
        if (sentence_index % 100) == 0:
            print('\t%s/%s' % (str(sentence_index), length_string))
    print('\t%s/%s' % (str(len(input_data)), length_string))

    dataY = np.zeros((len(input_data), sentence_length-1, len(dictionary)))
    for sentence_index, sentence in enumerate(dataY_init):
        dataY[sentence_index] = to_categorical(sentence, len(dictionary))

    return dataX, dataY


def setup(sentence_length, input_file_path):
    story_raw, int_to_word, word_to_int = read_input(input_file_path)
    story = generate_story(story_raw, sentence_length)
    dataX, dataY = encode(story, sentence_length, word_to_int)

    return dataX, dataY, story, word_to_int, int_to_word
