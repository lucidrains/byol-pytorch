from pathlib import Path

import fanova.visualizer
import hpbandster.core.result as hpres
import numpy as np

from fanova import fANOVA

from metassl.hyperparameter_optimization.configspaces import get_color_jitter_strengths_configspace


def get_fanova_plots(path, config_space):
    # Load the example run from the log files
    result = hpres.logged_results_to_HBS_result(path)

    # Save the used config space
    config_space = config_space
    file_to_write_in = open(path + "/config_space.txt", "w")
    file_to_write_in.write(
        "Used config space:" + "\n==================\n" + str(config_space) + "\n\n"
    )
    file_to_write_in.close()

    # fanova object
    X, Y, config_space = result.get_fANOVA_data(
        config_space, budgets=None, loss_fn=lambda r: r.loss, failed_loss=None
    )
    f = fANOVA(X, Y)

    # Generate hyperparameter importance file
    # TODO @Diane - Implement that

    # Generate folder with fanova visualization plots
    Path(path + "/fanova_plots").mkdir(parents=True, exist_ok=True)
    f.set_cutoffs(cutoffs=(-np.inf, np.inf))
    vis = fanova.visualizer.Visualizer(f, config_space, path + "/fanova_plots/")
    vis.create_all_plots()


if __name__ == "__main__":
    result_path = "/home/wagn3rd/Projects/metassl/results/diane/first_bohb"
    config_space = get_color_jitter_strengths_configspace()
    get_fanova_plots(path=result_path, config_space=config_space)