#!/bin/bash
python metassl/train_alternating_simsiam.py --config "metassl/default_metassl_config.yaml" --expt.expt_name $1 --train.epochs $2 --finetuning.epochs $3 --expt.warmup_epochs $4
