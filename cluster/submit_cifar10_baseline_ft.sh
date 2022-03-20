#!/bin/bash
#SBATCH -p testdlc_gpu-rtx2080  # mlhiwidlc_gpu-rtx2080
#SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J MSSL_D_Cifar10_SimSiam_FT
#SBATCH -t 00:30:00

source activate metassl

python -m metassl.train_linear_classifier_simsiam --config "metassl/default_metassl_config_cifar10.yaml" \
						  --use_fixed_args \
						  --data.dataset_percentage_usage 100 \
						  --finetuning.epochs 100 \
						  --expt.warmup_epochs 0 \
						  --expt.seed 0 \
						  --expt.save_model_frequency 10 \
						  --expt.is_non_grad_based \
						  --expt.multiprocessing_distributed \
						  --model.arch baseline_resnet \
						  --expt.expt_name $EXPERIMENT_NAME \
						  --expt.target_model_checkpoint_path "/work/dlclarge2/wagnerd-metassl-experiments/metassl/CIFAR10/22-03-14_solarize_seed0/checkpoint_0799.pth.tar"
						  # --expt.target_model_checkpoint_path "/work/dlclarge2/wagnerd-metassl-experiments/metassl/CIFAR10/22-03-08_solarize_seed6/checkpoint_0799.pth.tar"



