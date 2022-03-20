#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:8
#SBATCH -J MSSL_IN_SimSiam_FT
#SBATCH -t 23:59:59

python -m metassl.train_linear_classifier_simsiam --config "metassl/default_metassl_config_imagenet.yaml" \
						  --use_fixed_args \
						  --data.dataset_percentage_usage 100 \
						  --finetuning.epochs 100 \
						  --expt.warmup_epochs 0 \
						  --expt.seed 0 \
						  --expt.save_model_frequency 5 \
						  --expt.is_non_grad_based \
						  --expt.expt_name $EXPERIMENT_NAME \
						  --expt.ssl_model_checkpoint_path "/work/dlclarge2/wagnerd-metassl-experiments/metassl/ImageNet/22-03-08_baseline_ft_seed0/linear_cls_0060.pth.tar"



