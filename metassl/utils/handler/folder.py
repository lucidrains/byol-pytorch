import pathlib

from metassl.utils.handler.base_handler import Handler

"""
Handle the location, new folders and experiments sub-folder structure.

base_dir / project / session / experiment

experiment will be increased

"""


class FolderHandler(Handler):

    def __init__(self, experiments_dir, project_name=None, session_name=None, experiment_name=None, count_expt=False,
                 reload_expt=False):
        super().__init__()

        self.experiments_dir = pathlib.Path(experiments_dir)

        # self.subfolder = ["log", "checkpoint", "config", "profile"]

        if project_name is not None:
            self.project_name = project_name
            self.session_name = session_name
            self.experiment_name = experiment_name
            self.count_expt = count_expt
            self.reload_expt = reload_expt

            self.expt_dir = self.create_folder()
        else:
            self.expt_dir = self.experiments_dir

    def create_folder(self):

        dir = self.experiments_dir
        self.save_mkdir(dir)

        for folder in [self.project_name, self.session_name]:
            dir = dir / folder
            self.save_mkdir(dir)


        if self.reload_expt:
            self.experiment_name = self.get_latest_name(dir, self.experiment_name)
        elif self.count_expt:
            self.experiment_name = self.counting_name(dir, self.experiment_name)

        dir = dir / self.experiment_name
        self.save_mkdir(dir)

        # for folder in self.subfolder:
        #     self.save_mkdir(dir / folder)

        return dir

    @property
    def dir(self):
        return self.expt_dir
