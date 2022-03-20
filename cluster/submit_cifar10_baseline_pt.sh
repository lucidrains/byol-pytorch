#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J MSSL_D_Cifar10_SimSiam
#SBATCH -t 23:00:00

source activate metassl

python -m metassl.train_simsiam --config "metassl/default_metassl_config_cifar10.yaml" \
				--use_fixed_args \
				--data.dataset_percentage_usage 100 \
				--train.epochs 800 \
				--expt.warmup_epochs 0 \
				--expt.seed 2 \
				--expt.save_model_frequency 50 \
				--expt.is_non_grad_based \
				--expt.multiprocessing_distributed \
				--model.arch "baseline_resnet" \
				--simsiam.use_baselines_loss \
				--expt.expt_name $EXPERIMENT_NAME
				# --expt.use_fix_aug_params


