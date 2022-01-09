from hpbandster.core.worker import Worker

from metassl.train_alternating_simsiam import main

class HPOWorker(Worker):
    def __init__(self, yaml_config, expt_dir, **kwargs):
        self.yaml_config = yaml_config
        self.expt_dir = expt_dir
        super().__init__(**kwargs)

    def compute(self, config_id, config, budget, working_directory, *args, **kwargs):
        bohb_infos = {'bohb_config_id': config_id, 'bohb_config': config, 'bohb_budget': budget}
        val_metric = main(config=self.yaml_config, expt_dir=self.expt_dir, bohb_infos=bohb_infos)
        return {
            "loss": -1 * val_metric,
            "info": {"test/metric": 0},
        }  # remember: HpBandSter always minimizes!
