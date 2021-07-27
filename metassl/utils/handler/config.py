import os
import yaml
import pathlib

from boho.utils.handler.base_handler import Handler

"""
reads a yml config or a dict and safes it into experiment folder
"""


class AttributeDict(Handler):
    def __init__(self, dictionary, name):
        super().__init__()

        # if not hasattr(self, "config_attr"):
        #     self.config_attr = []

        for key in dictionary:
            if isinstance(dictionary[key], dict):
                if not hasattr(self, "sub_config"):
                    self.sub_config = []
                self.sub_config.append(key)
                setattr(self, key, AttributeDict(dictionary[key], key))
            else:
                # self.config_attr.append(key)
                setattr(self, key, dictionary[key])

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    @property
    def get_dict(self):
        return self.__dict__

    @property
    def dict(self):
        return self.__dict__

    def set_attr(self, name, value):
        if isinstance(value, pathlib.Path):
            value = value.as_posix()
        self.__setattr__(name, value)


class ConfigHandler(AttributeDict):

    def __init__(self, config_file=None, config_dict=None):

        if config_file is None and config_dict is None:
            raise UserWarning("ConfigHandler: config_file and config_dict is None")

        elif config_file is not None and config_dict is None:
            with open(config_file, 'r') as f:
                config_dict = yaml.load(f, Loader=yaml.Loader)

        super().__init__(config_dict, "main")

        self.check_experiment_config()

    def check_experiment_config(self):
        if hasattr(self, "expt"):
            for attr_name in ['project_name', 'session_name', 'experiment_name']:
                if not hasattr(self.expt, attr_name):
                    raise UserWarning(f"ConfigHandler: {attr_name} is missing")
                elif isinstance(self.expt.__getattribute__(attr_name), str):
                    self.expt.__setattr__(attr_name, str(self.expt.__getattribute__(attr_name)))

    def save_config(self, dir, file_name="config.yml"):
        dir = pathlib.Path(dir)
        self.save_mkdir(dir)
        if os.path.isfile(dir / file_name):
            file_name = self.counting_name(dir, file_name, suffix=True)
        with open(dir / file_name, 'w+') as f:
            config_dict = self.get_dict
            yaml.dump(config_dict, f, default_flow_style=False, encoding='utf-8')
        return dir / file_name
#
# if __name__ == "__main__":
#     config_dir = "/home/joerg/workspace/python/gitlab_projects/workbench/test/workbench/utils/handler/dummy_config.yml"
#
#
#     config = ConfigHandler(config_dir=config_dir)
#
#     print(config.seed)
#     print(config.ga.mlp)
#
#     config.seed += 23
#
#     dirr = config.save_config("development")
#
#     print(config.get_dict)
#
#     print("2222222222222")
#
#     config2 = ConfigHandler(config_dir=dirr)
#
#     print(config2.seed)
#     print(config2.ga.mlp)
#
#
#     config_dir = "development/actor_critic_nevo/dev_config.yml"
#
#     with open(pathlib.Path(os.getcwd()).parents[2] / config_dir, 'r') as f:
#         config_dict = yaml.load(f, Loader=yaml.Loader)
#
#     cfg = ConfigHandler(config_dict=config_dict)
#
#     def dummy_model(num_input,  num_outputs, hidden_size,
#                                            activation,
#                                            output_activation,
#                                            layer_norm,
#                                            output_vanish):
#
#        print("num_input", num_input)
#        print("num_outputs", num_outputs)
#        print("hidden_size", hidden_size)
#        print("activation", activation)
#        print("output_activation", output_activation)
#        print("layer_norm", layer_norm)
#        print("output_vanish", output_vanish)
#
#     print(cfg.get_dict)
#     print(str(cfg.get_dict))
#     dummy_model(num_input=2, num_outputs=3, **cfg.actor.get_dict)
