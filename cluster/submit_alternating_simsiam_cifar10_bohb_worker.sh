#!/bin/bash
#SBATCH -q dlc-wagnerd # partition (queue)
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -t 0-06:00 # time (D-HH:MM)
#SBATCH --gres=gpu:1
#SBATCH -J mssl-w-cifar10-simsiam # sets the job name. If not specified, the file name will be used as job name
#SBATCH --array 0-70%10
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
echo "python_version $(which python)"
source activate ssl
echo "python_version $(which python)"
# Job to perform
python -m metassl.train_alternating_simsiam --expt.expt_name $EXPERIMENT_NAME \
	--config "metassl/default_metassl_config_cifar10.yaml" \
	--expt.expt_mode "CIFAR10_BOHB" \
	--expt.is_non_grad_based \
	--use_fixed_args \
	--finetuning.valid_size 0.1 \
	--train.epochs 450 \
	--finetuning.epochs 450 \
	--expt.warmup_epochs 0 \
	--data.dataset_percentage_usage 25 \
	--bohb.configspace_mode "probability_augment" \
	--expt.data_augmentation_mode "probability_augment" \
	--finetuning.data_augmentation "none" \
	--bohb.max_budget 450 \
	--bohb.min_budget 450 \
	--bohb.n_iterations 70 \
	--bohb.worker \
	--bohb.run_id "probability_augment_pt-only"

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
