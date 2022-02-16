#!/bin/bash
python metassl/train_alternating_simsiam_parameterized_aug_nn_module.py --config "metassl/default_metassl_config_cifar10.yaml" --use_fixed_args --expt.expt_name $1 --train.epochs $2 --expt.warmup_epochs $3
