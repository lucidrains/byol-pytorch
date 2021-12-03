#!/bin/bash
python metassl/train_alternating_simsiam.py --config "metassl/default_metassl_config.yaml" --expt.expt_name $1 --train.epochs $2 --finetuning.epochs $3 --train.lr $4 --finetuning.lr $5 --model.turn_off_bn
