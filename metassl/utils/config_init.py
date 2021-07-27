from typing import Dict, List
import inspect
from metassl.utils.handler.config import ConfigHandler, AttributeDict


def cinit(instance, config, **kwargs):
    """
    Instantiates a class by selecting the required args from a ConfigHandler. Omits wrong kargs
    @param instance:    class
    @param config:      ConfigHandler object contains class args
    @param kwargs:      kwargs besides/replacing ConfigHandler args
    @return:            class object
    """



    if isinstance(instance, type):
        instance_args = inspect.signature(instance.__init__)
        instance_keys = list(instance_args.parameters.keys())
        instance_keys.remove("self")
    else:
        instance_keys = inspect.getfullargspec(instance).args

    if isinstance(config, ConfigHandler) or isinstance(config, AttributeDict):
        config_dict = config.get_dict
    elif isinstance(config, Dict):
        config_dict = config
    elif isinstance(config, List):
        config_dict = {}
        for sub_conf in config:
            if isinstance(sub_conf, ConfigHandler) or isinstance(sub_conf, AttributeDict):
                config_dict.update(sub_conf.get_dict)
            elif isinstance(sub_conf, Dict):
                config_dict.update(sub_conf)
    else:
        raise UserWarning(f"cinit: Unknown config type. config must be Dict, AttributeDict or ConfigHandler but is {type(config)}")

    init_dict = {}

    for name, arg in kwargs.items():
        if name in instance_keys:
            init_dict[name] = arg

    for name, arg in config_dict.items():
        if name in instance_keys and name not in init_dict.keys():
            init_dict[name] = arg

    init_keys = list(init_dict.keys())
    missing_keys = list(set(instance_keys) - set(init_keys))
    if len(missing_keys) > 0:
        raise UserWarning(f"cinig: keys missing {missing_keys}")

    return instance(**init_dict)


if __name__ == "__main__":
    econfig = ConfigHandler(config_dict={"optimizer": {"lr": 123, "alpha": 0.3, "beta": 0.4, "kappa": 1.2},
                                         "expt": {'project_name': "", 'session_name': "", 'experiment_name': ""}})


    class ExampleClass():
        def __init__(self, lr, alpha, kappa):
            self.lr, self.alpha, self.kappa = lr, alpha, kappa

        def print(self):
            print(f"class lr {self.lr}, alpha {self.alpha}, kappa {self.kappa}")


    eclass = cinit(ExampleClass, econfig.optimizer)
    eclass.print()

    def example_method(lr, alpha, kappa=0.2):
            return f"method lr {lr}, alpha {alpha}, kappa {kappa}"


    eclass = cinit(example_method, econfig.optimizer)
    print(eclass)