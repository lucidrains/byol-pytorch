#!/bin/bash
python metassl/train_simsiam.py --expt.expt_name $1 --train.epochs $2 --finetuning.epochs $3 --expt.warmup_epochs $4 --expt.wd_decay_pt --train.wd_start 4e-2 --train.wd_end 4e-1 --config $5 --expt.expt_mode "ImageNet" --data.dataset "ImageNet"  --use_fixed_args
python metassl/train_linear_classifier_simsiam.py --expt.expt_name $1 --train.epochs $2 --finetuning.epochs $3 --expt.warmup_epochs $4 --expt.wd_decay_pt --config $5 --expt.expt_mode "ImageNet" --data.dataset "ImageNet"  --use_fixed_args
