#!/usr/bin/env bash
python metassl/train_simsiam.py --expt_name $1 --epochs $2 --lr $3 --ssl_model_checkpoint_path $4 
