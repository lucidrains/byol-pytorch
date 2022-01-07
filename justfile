# List available just commands
@list:
  just --list

# ---------------------------------------------------------------------------------------
# SIMSIAM ON CIFAR10
# ---------------------------------------------------------------------------------------

# Submit alternating SimSiam on CIFAR10 with default setting to reproduce results
@simsiam_cifar10_default EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --nodelist=dlcgpu14 --output=/work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_cifar10_default.sh

# Submit alternating SimSiam on CIFAR10 with some experimental settings
@simsiam_cifar10_workspace EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --nodelist=dlcgpu13 --output=/work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_cifar10_workspace.sh

# ---------------------------------------------------------------------------------------
# SIMSIAM ON CIFAR10 WITH BOHB
# ---------------------------------------------------------------------------------------

# Submit master to train alternating SimSiam on CIFAR10 with BOHB
@simsiam_cifar10_master EXPERIMENT_NAME:
  #!/usr/bin/env bash
  python -m metassl.train_alternating_simsiam --expt.expt_name {{EXPERIMENT_NAME}} --config "metassl/default_metassl_config_cifar10.yaml" --expt.expt_mode "CIFAR10_BOHB" --expt.is_non_grad_based --use_fixed_args --finetuning.valid_size 0.1 --train.epochs 450 --finetuning.epochs 450 --expt.warmup_epochs 0 --data.dataset_percentage_usage 25 --bohb.nic_name "enp1s0"

# Submit worker to train alternating SimSiam on CIFAR10 with BOHB
@simsiam_cifar10_worker EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_cifar10_bohb_worker.sh

# ---------------------------------------------------------------------------------------
# SIMSIAM ON IMAGENET
# ---------------------------------------------------------------------------------------

# Submit alternating SimSiam on IMAGENET with default setting to reproduce results
@simsiam_imagenet_default EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --nodelist=dlcgpu14 --output=/work/dlclarge2/wagnerd-metassl_experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_imagenet_default.sh

# Submit alternating SimSiam on IMAGENET with some experimental settings
@simsiam_imagenet_workspace EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl_experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_imagenet_workspace.sh

# ---------------------------------------------------------------------------------------
# SIMSIAM ON IMAGENET WITH BOHB
# ---------------------------------------------------------------------------------------

# Submit master to train alternating SimSiam on IMAGENET with BOHB
@simsiam_imagenet_master EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_imagenet_bohb_master.sh

# Submit worker to train alternating SimSiam on IMAGENET with BOHB
@simsiam_imagenet_worker EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/ImageNet/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_imagenet_bohb_worker.sh
