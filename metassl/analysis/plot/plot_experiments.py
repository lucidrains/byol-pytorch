from typing import List
import os, sys
import json
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from metassl.analysis.plot.find_experiments import find_experiments
from metassl.analysis.plot.load_experiment import load_experiment

user = os.environ.get('USER')
expt_dir = f"/home/{user}/workspace/experiments/boho"

min_len = 1

# rsync -av --progress --exclude *.pth  frankej@aadlogin.informatik.uni-freiburg.de:/mhome/frankej/workspace/experiments/boho/*  /home/joerg/workspace/experiments/cluster/boho/
# rsync -av --progress --exclude *.pth  frankej@aadlogin.informatik.uni-freiburg.de:/mhome/frankej/workspace/experiments/boho/*  /home/frankej/workspace/experiments/cluster/boho/

experiment = "boho"

rlow_lists = [
                    ['boho_baseline_3_small_resnet', "grid",   ],
                    ['boho_baseline_3_small_resnet', "random",   ],
                      ]

summary_keys =[
                 'train_loss',
                 'valid_accuracy',
                 'learning_rate',
                 ]



ALL_IN_ONE = False


expt_list = [find_experiments(expt_dir, keys) for keys in rlow_lists]

print(f"found {len(expt_list)} experiments")
print(expt_list)

## print experiment dirs
for idx, expt_l in enumerate(expt_list):
    print(f"### key list {idx}:")
    for expt_d in expt_l:
        print(expt_d['dir'].__str__().split("/boho/")[1])

expt_list = [[load_experiment(expt_d, summary_keys) for expt_d in  expt_l] for expt_l in expt_list]


print(f"load {len(expt_list)} experiments")

colormap = plt.cm.nipy_spectral  # nipy_spectral, Set1,Paired

if ALL_IN_ONE:
    a = [len([e for e in l if e]) for l in expt_list]
    colors = [colormap(i) for i in np.linspace(0, 1, sum(a))]
    all_idx = 0

plt.rcParams.update({'font.size': 5})
fig1 = plt.figure(figsize=(30, 20), dpi=200)

for c_id, expt_l in enumerate(expt_list):
    if ALL_IN_ONE:
        c_id = 0
    else:
        colors = [colormap(i) for i in np.linspace(0, 1, len(expt_l))]


    for e_idx, expt_d in enumerate(expt_l):

        if ALL_IN_ONE:
            e_idx = all_idx
            all_idx += 1

        if expt_d:


            for k_id, key in enumerate(summary_keys):

                if ALL_IN_ONE:
                    ax1 = fig1.add_subplot(len(summary_keys), 1, (k_id)  + (1 + c_id))
                else:
                    ax1 = fig1.add_subplot(len(summary_keys), len(rlow_lists), (k_id) * len(rlow_lists) + (1 + c_id))

                print(key)

                # try:
                ax1.set_ylabel(key)

                expt_name = expt_d['expt_name'].split("/")[1]

                p = ax1.plot(expt_d[key]['step'][:-1], expt_d[key]['value'], label=expt_name, color=colors[e_idx])

                if e_idx % 2 == 0:
                    p[0].set_linestyle("--")

                if np.max(expt_d[key]['value']) - np.min(expt_d[key]['value']) > 1000:
                    ax1.set_yscale('log')

                ax1.grid(True)

                if "_test" in key and "solved" in key:
                    ax1.set_ylim(0,0.2)
                elif "_test" in key:
                    ax1.set_ylim(0, 0.8)

                # except:
                #     pass

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    # ax1.legend(loc='upper center', bbox_to_anchor=(0.9, -0.05),
    #           fancybox=True, shadow=True, ncol=1)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=2)

# fig1.set_size_inches(16,40,forward=True)
plt.subplots_adjust(top=0.98, bottom=0.25)

# plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.45)
plt.savefig(f"{expt_dir}/{rlow_lists[0][0]}.png")
plt.show()
