#!/bin/bash
set -e
set -x

python evaluate.py data/model_adam_64.h5 data/test.conll
python evaluate.py data/model_adagrad_64.h5 data/test.conll
python evaluate.py data/model_sgd_64.h5 data/test.conll
python evaluate.py data/model_sm3_64.h5 data/test.conll

python evaluate.py data/model_adam_512.h5 data/test.conll
python evaluate.py data/model_adagrad_512.h5 data/test.conll
python evaluate.py data/model_sgd_512.h5 data/test.conll
python evaluate.py data/model_sm3_512.h5 data/test.conll
