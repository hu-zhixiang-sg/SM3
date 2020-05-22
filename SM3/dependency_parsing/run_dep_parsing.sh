#!/bin/bash
set -e
set -x

python get_vocab.py data/train.conll data/words.vocab data/pos.vocab
python extract_training_data.py data/train.conll data/input_train.npy data/target_train.npy

python train_model.py data/input_train.npy data/target_train.npy adam 64
python train_model.py data/input_train.npy data/target_train.npy adagrad 64
python train_model.py data/input_train.npy data/target_train.npy sgd 64
python train_model.py data/input_train.npy data/target_train.npy sm3 64

python train_model.py data/input_train.npy data/target_train.npy adam 512
python train_model.py data/input_train.npy data/target_train.npy adagrad 512
python train_model.py data/input_train.npy data/target_train.npy sgd 512
python train_model.py data/input_train.npy data/target_train.npy sm3 512
