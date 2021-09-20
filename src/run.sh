#! /bin/env bash
seed=1234
n_cnn=2
val=0.2
n_fc=3
ok_weight=0.2
lr=0.001
python3 run.py \
    -d ../data \
    -lr $lr \
    --read-mode gray \
    --batch-size 64 \
    --seed $seed \
    --num-cnn $n_cnn \
    --num-fc $n_fc \
    --epoch 10 \
    --class-num 2 \
    --val-data $val\
    --ok-weight $ok_weight \
    --binary-image \
    -o ../output/