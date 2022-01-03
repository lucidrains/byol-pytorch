# List available just commands
@list:
  just --list

# ---------------------------------------------------------------------------------------
# SIMSIAM ON CIFAR10
# ---------------------------------------------------------------------------------------

# Submit alternating SimSiam on CIFAR10 with default setting to reproduce results
@simsiam_cifar10_default EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --nodelist=dlcgpu14 --output=experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_cifar10_default.sh

# Submit alternating SimSiam on CIFAR10 with some experimental settings
@simsiam_cifar10_workspace EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --nodelist=dlcgpu14 --output=experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_cifar10_workspace.sh

# ---------------------------------------------------------------------------------------
# SIMSIAM ON CIFAR10 WITH BOHB
# ---------------------------------------------------------------------------------------

# Submit master to train alternating SimSiam on CIFAR10 with BOHB
@simsiam_cifar10_master EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_cifar10_bohb_master.sh

# Submit worker to train alternating SimSiam on CIFAR10 with BOHB
@simsiam_cifar10_worker EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_cifar10_bohb_worker.sh

# ---------------------------------------------------------------------------------------
# SIMSIAM ON IMAGENET
# ---------------------------------------------------------------------------------------

# Submit alternating SimSiam on IMAGENET with default setting to reproduce results
@simsiam_imagenet_default EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p experiments/IMAGENET/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --nodelist=dlcgpu14 --output=experiments/IMAGENET/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=experiments/IMAGENET/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_imagenet_default.sh

# Submit alternating SimSiam on IMAGENET with some experimental settings
@simsiam_imagenet_workspace EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p experiments/IMAGENET/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --nodelist=dlcgpu14 --output=experiments/IMAGENET/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=experiments/IMAGENET/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_imagenet_workspace.sh

# ---------------------------------------------------------------------------------------
# SIMSIAM ON IMAGENET WITH BOHB
# ---------------------------------------------------------------------------------------

# Submit master to train alternating SimSiam on IMAGENET with BOHB
@simsiam_imagenet_master EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p experiments/IMAGENET/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=experiments/IMAGENET/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=experiments/IMAGENET/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_imagenet_bohb_master.sh

# Submit worker to train alternating SimSiam on IMAGENET with BOHB
@simsiam_imagenet_worker EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p experiments/IMAGENET/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=experiments/IMAGENET/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=experiments/IMAGENET/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_alternating_simsiam_imagenet_bohb_worker.sh