from typing import List, Dict, Tuple
import math
import numpy as np
import torch

import matplotlib.pyplot as plt


class SummaryDict():
    """
    Similar to TensorFlow summary but can deal with lists, stores everything in numpy arrays. Please see main for usage.
    """
    def __init__(self, summary=None):
        self.summary = {}
        if summary is not None:
            for key, value in summary.items():
                self.summary[key] = value

    @property
    def keys(self):
        keys = list(self.summary.keys())
        if "step" in keys:
            keys.remove("step")
        return keys

    def __call__(self, summary):

        if isinstance(summary, SummaryDict):
            for key, value_lists in summary.summary.items():
                if key in self.summary.keys():
                    if key == "step":
                        if min(value_lists) != 1 + max(self.summary['step']):
                            value_lists = np.asarray(value_lists) + max(self.summary['step']) + 1 - min(value_lists)
                    self.summary[key] = np.concatenate([self.summary[key], value_lists], axis=0)
                else:
                    self.summary[key] = value_lists

        elif isinstance(summary, Dict):
            for name, value in summary.items():
                self.__setitem__(name, value)
        elif isinstance(summary, (Tuple, List)):
            for l in summary:
                self.__call__(l)
        else:
            raise UserWarning(f"SummaryDict: call not implementet for type: {type(summary)}")

    def __setitem__(self, key, item):

        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        if isinstance(item, List):
            if isinstance(item[0], torch.Tensor):
                item = [v.cpu().numpy() for v in item]
        if isinstance(item, np.ndarray):
            item = np.squeeze(item)

        item = np.expand_dims(np.asarray(item), axis=0)
        if item.shape.__len__() < 2:
            item = np.expand_dims(item, axis=0)

        if key not in self.summary.keys():
            self.summary[key] = item
        else:
            self.summary[key] = np.concatenate([self.summary[key], item], axis=0)

    def __getitem__(self, key):
        return self.summary[key]

    def save(self, file):
        np.save(file, self.summary)

    def load(self, file):
        return np.load(file).tolist()


def plot_summary(summary, fig_num=1, kappa=0, plot_keys=None):
    if plot_keys is None:
        plot_keys = summary.keys

    n_plots = len(plot_keys)
    if n_plots <= 3:
        plot_size = (n_plots, 1)
    elif n_plots <= 6:
        plot_size = (math.ceil(n_plots / 2), 2)
    elif n_plots <= 12:
        plot_size = (math.ceil(n_plots / 3), 3)
    elif n_plots <= 16:
        plot_size = (math.ceil(n_plots / 4), 4)
    else:
        raise UserWarning(f"plot summary: to many keys, choose 16: {plot_keys}")

    fig = plt.figure(num=fig_num)
    plt.clf()
    ax_list = [fig.add_subplot(*plot_size, i + 1) for i in range(len(plot_keys))]

    step = summary['step']
    for idx, key in enumerate(plot_keys):
        value = summary[key]
        ax_list[idx].plot(step, value)
        ax_list[idx].set_ylabel(key)
        # if np.min(value) >= 0:
        if (np.max(value) - np.min(value)) > 20 and np.min(value) >= 0:
            ax_list[idx].set_yscale('log')

        if key == 'ma_ce_loss':
            ax_list[idx].hlines(kappa, xmin=0, xmax=max(step), color='r')

    ax_list[idx].set_xlabel('Steps')

    # plt.show()
    plt.pause(0.001)


if __name__ == "__main__":

    my_summary = SummaryDict()

    for i in range(100):
        my_summary["step"] = i
        a = torch.LongTensor([i]* (i+1))
        print("a", a.shape)
        my_summary["A"] = a
        my_summary["B"] = i ** 2
        my_summary["C"] = [i, i * 2, i * 3]

    my_2_summary = SummaryDict()

    for i in range(80, 120):
        my_2_summary["step"] = i
        my_2_summary["A"] = i
        my_2_summary["B"] = i ** 2
        my_2_summary["C"] = [i, i * 2, i * 3]

    my_summary(my_2_summary)
    # my_summary = my_2_summary

    for key, value in my_summary.summary.items():
        print(key, value.shape)

    plot_summary(my_summary)

    dir = "/home/joerg/workspace/experiments/cluster/rna/lerna_factor_2/"
    file = dir + "rna_encoder_9_loss-geco_mdim-512_clip_grad-0.1-f-50_seed-8262-0/checkpoint/train_summary.npy"  # nan
    file = dir + "rna_encoder_6_loss-geco_mdim-512_clip_grad-0.1-f-5_seed-8262-0/checkpoint/train_summary.npy"
    file = dir + "rna_encoder_7_loss-geco_mdim-512_clip_grad-0.1-f-10_seed-8262-0/checkpoint/train_summary.npy"
    # file  = dir+"rna_encoder_8_loss-geco_mdim-512_clip_grad-0.1-f-20_seed-8262-0/checkpoint/train_summary.npy" #nan

    file = "/home/joerg/workspace/experiments/cluster/rna/toyseqdist_decoder_expt_1/toyseq_encoder_7_geco_transformer-128_toy-arbitrary-small_prior-False_kappa-0.05_layer-all_seed-4852-0/checkpoint/train_summary.npy"

    summary = np.load(file).tolist()

    # plot_summary(summary.keys(), summary)
    # input("Press Enter to continue...")
    # plot_summary(summary.keys(), summary)
    # input("Press Enter to continue...")

    summary.save("test.npy")
    summary.load("test.npy")

    plot_keys = summary.keys

    print("plot_keys: ", plot_keys)

    fig_num = 1

    n_plots = len(plot_keys)
    if n_plots <= 3:
        plot_size = (n_plots, 1)
    elif n_plots <= 6:
        plot_size = (math.ceil(n_plots / 2), 2)
    elif n_plots <= 12:
        plot_size = (math.ceil(n_plots / 3), 3)
    elif n_plots <= 16:
        plot_size = (math.ceil(n_plots / 4), 4)
    else:
        raise UserWarning(f"plot summary: to many keys, choose 16: {plot_keys}")

    # fig = plt.figure(num=fig_num)
    # plt.clf()
    # ax_list = [fig.add_subplot(*plot_size, i + 1) for i in range(len(plot_keys))]
    ax_list = [plt.subplot(*plot_size, i + 1) for i in range(len(plot_keys))]

    for idx, key in enumerate(plot_keys):

        if not isinstance(summary[key], List):
            # value = np.stack(summary[key], axis=0)
            # value = np.nan_to_num(value)
            # step = np.stack(summary['step'], axis=0)

            value = summary[key]
            step = summary['step']
            # print(np.stack(summary[key], axis=0))
            print("key", key)
            print("value", value.shape)
            print("step", step.shape)

            ax_list[idx].plot(step, value)
            # ax_list[idx].plot(summary['step'], summary[key])
            ax_list[idx].set_ylabel(key)
            if max(value) > 20 and min(value) > 0:
                ax_list[idx].set_yscale('log')

            # if key == 'ma_ce_loss':
            #     ax_list[idx].hlines(0.05, xmin=0, xmax=max(summary['step']), color='r')
        else:

            value_list = list(map(list, zip(*summary[key])))

            for value in value_list:
                # print("list_value: ", value)
                # print("list_value: ", value.__len__())
                # value = np.stack(value, axis=0)
                # value = np.nan_to_num(value)
                # step = np.stack(summary['step'], axis=0)
                # print("value: ", value.shape)
                # print("step: ", step.shape)

                # value = summary[key]
                step = summary['step']

                print("key", key)
                print("value", value.shape)
                print("step", step.shape)

                ax_list[idx].plot(step, value)
                # ax_list[idx].plot(summary['step'], summary[key])
                ax_list[idx].set_ylabel(key)
                if max(value) > 10 and min(value) > 0:
                    ax_list[idx].set_yscale('log')

    ax_list[idx].set_xlabel('Steps')

    # plt.show()
    # plt.pause(0.001)
