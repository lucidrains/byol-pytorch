#!/bin/bash
#SBATCH -q dlc-wagnerd # partition (queue)
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH --gres=gpu:1
#SBATCH -J mssl-d-cifar10-simsiam # sets the job name. If not specified, the file name will be used as job name
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Job to perform
python -m metassl.train_alternating_simsiam --expt.expt_name $EXPERIMENT_NAME --config "metassl/default_metassl_config_imagenet.yaml" --expt.expt_mode "IMAGENET_BOHB" --expt.is_non_grad_based --use_fixed_args --finetuning.valid_size 0.1 --train.epochs 90 --finetuning.epochs 90 --expt.warmup_epochs 0

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";