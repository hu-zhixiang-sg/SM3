"""
Trains model with given optimizer and saves to data/<model_file>
Sync with Weights and Biases (wandb) to track metrics (loss, accuracy, CPU, memory, etc.)
"""

from extract_training_data import FeatureExtractor
import os
import sys
import numpy as np
import keras
from keras import Sequential
from keras.layers import Flatten, Embedding, Dense

import pickle

os.sys.path.insert(0, os.path.abspath('../'))
from sm3i.sm3 import sm3

import wandb
from wandb.keras import WandbCallback
wandb.init(project="sm3")


def build_model(word_types, pos_types, outputs, lr=0.01, optimizer=keras.optimizers.Adam):
    if not callable(optimizer):
        if optimizer == 'sm3':
            optimizer = sm3.SM3Optimizer
        if optimizer == 'adam':
            optimizer = keras.optimizers.Adam
        elif optimizer == 'nadam':
            optimizer = keras.optimizers.Nadam
        elif optimizer == 'adagrad':
            optimizer = keras.optimizers.Adagrad
        elif optimizer == 'rmsprop':
            optimizer = keras.optimizers.RMSprop
        else:
            optimizer = keras.optimizers.SGD

    model = Sequential()

    model.add(Embedding(word_types, 32, input_length=6))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(outputs, activation='softmax'))
    model.compile(optimizer(learning_rate=lr), loss="categorical_crossentropy",
                  metrics=['categorical_accuracy'])

    return model


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE, 'r')
        pos_vocab_f = open(POS_VOCAB_FILE, 'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    optimizer, batch_size = sys.argv[3].lower(), int(sys.argv[4])
    model_name = "model_{}_{}.h5".format(optimizer, batch_size)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    print("Compiling model.")
    model = build_model(len(extractor.word_vocab), len(extractor.pos_vocab),
                        len(extractor.output_labels), optimizer=optimizer)
    inputs = np.load(sys.argv[1])
    outputs = np.load(sys.argv[2])
    print("Done loading data.")

    # Now train the model
    history = model.fit(inputs, outputs, epochs=10, batch_size=batch_size, callbacks=[WandbCallback()])

    model.save(os.path.join('data', model_name))
    model.save(os.path.join(wandb.run.dir, model_name))
