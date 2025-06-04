#!/bin/bash
set -e
DATASET=$1
MODEL=$2
python models/${MODEL}/test.py --dataset data/${DATASET} --model ${MODEL}
