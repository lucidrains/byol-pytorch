# Code based on: https://automl.github.io/HpBandSter/build/html/auto_examples/plot_example_6_analysis.html
import os
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
import matplotlib.pyplot as plt

def make_analysis(path):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(path)

    ######################################################################################
    # RESULTS
    ######################################################################################
    # For more see:
    # https://github.com/automl/HpBandSter/blob/master/hpbandster/core/result.py

    # get all executed runs
    all_runs = result.get_all_runs()

    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()
    print("RESULT", result)

    # Here is how you get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()

    # We have access to all information: the config, the loss observed during
    # optimization, and all the additional information
    print("HERE:", id2conf, inc_id)
    inc_config = id2conf[inc_id]["config"]

    print("Best found configuration:")
    print(inc_config)

    # Save best configuration
    file_to_write_in = open(path + "/results_loss_and_info.txt", "w")
    file_to_write_in.write("Best found configuration:" + "\n" + str(inc_config) + "\n\n")
    file_to_write_in.close()

    ######################################################################################
    # PLOTS
    ######################################################################################

    plots_folder = os.mkdir(path + "/bohb_plots")  # noqa: F841
    path_plots = path + "/bohb_plots"

    # Observed losses grouped by budget
    hpvis.losses_over_time(all_runs)
    plt.savefig(path_plots + "/losses_over_time.png")

    hpvis.concurrent_runs_over_time(all_runs)
    plt.savefig(path_plots + "/concurrent_runs_over_time.png")

    hpvis.finished_runs_over_time(all_runs)
    plt.savefig(path_plots + "/finished_runs_over_time.png")

    # Plot visualizes the spearman rank correlation coefficients of the losses between
    # different budgets.
    hpvis.correlation_across_budgets(result)
    plt.savefig(path_plots + "/correlation_across_budgets.png")

    # For model based optimizers, one might wonder how much the model actually helped.
    # Plot compares the performance of configs picked by the model vs. random ones
    hpvis.performance_histogram_model_vs_random(all_runs, id2conf)
    plt.savefig(path_plots + "/performance_histogram_model_vs_random.png")


if __name__ == "__main__":
    result_path = "/home/wagn3rd/Projects/metassl/results/diane/first_bohb"
    make_analysis(path=result_path)