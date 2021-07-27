import pathlib
from metassl.utils.handler.config import ConfigHandler
from metassl.utils.logger import Logger
from metassl.utils.handler.folder import FolderHandler
from metassl.utils.handler.checkpoint import CheckpointHandler


class Supporter():

    def __init__(self, experiments_dir=None, config_dir=None, config_dict=None, count_expt=False, reload_expt=False):

        if reload_expt:
            experiments_dir = pathlib.Path(experiments_dir)

            self.cfg = ConfigHandler(config_dir, config_dict)

            self.folder = FolderHandler(experiments_dir, self.cfg.expt.project_name, self.cfg.expt.session_name,
                                        self.cfg.expt.experiment_name, count_expt, reload_expt)
        else:
            self.cfg = ConfigHandler(config_dir, config_dict)

            if experiments_dir is None and self.cfg.expt.experiments_dir is None:
                raise UserWarning("ConfigHandler: experiment_dir and config.expt.experiment_dir is None")
            elif experiments_dir is not None:
                self.cfg.expt.set_attr("experiments_dir", experiments_dir)
            else:
                experiments_dir = pathlib.Path(self.cfg.expt.experiments_dir)

            self.folder = FolderHandler(experiments_dir, self.cfg.expt.project_name, self.cfg.expt.session_name,
                                        self.cfg.expt.experiment_name, count_expt)
        self.cfg.expt.experiment_name = self.folder.experiment_name
        self.cfg.expt.experiment_dir = self.folder.dir
        self.cfg.save_config(self.folder.dir)

        self.logger = Logger(self.folder.dir)
        self.ckp = CheckpointHandler(self.folder.dir)

        self.logger.log("project_name", self.cfg.expt.project_name)
        self.logger.log("session_name", self.cfg.expt.session_name)
        self.logger.log("experiment_name", self.cfg.expt.experiment_name)

    def get_logger(self):
        return self.logger

    def get_config(self):
        return self.cfg

    def get_checkpoint_handler(self):
        return self.ckp


if __name__ == "__main__":
    experiments_dir = "/home/joerg/workspace/python/gitlab_projects/workbench/development/"
    config_dir = "/home/joerg/workspace/python/gitlab_projects/workbench/test/workbench/utils/handler/dummy_config.yml"

    with Supporter(experiments_dir, config_dir=config_dir, count_expt=False) as sup:
        cfg = sup.get_config()
        print(cfg.ga.mlp)

        log = sup.get_logger()
        log.log("START")
        log.log("performance", 234.45)
        log.dump(key="21313", value=[2342343242, 234324324, 234324324])

    print("train end")
