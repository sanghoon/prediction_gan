#!/usr/bin/env bash

# Run without prediction
python example_dcgan.py --dataset cifar10 --dataroot cifar10 --cuda --lr 0.001 --outf out_vanilla

# Run with prediction
python example_dcgan.py --dataset cifar10 --dataroot cifar10 --cuda --lr 0.001 --outf out_predicition --pred