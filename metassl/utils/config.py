import jsonargparse
import yaml

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        def from_nested_dict(data):
            """ Construct nested AttrDicts from nested dictionaries. """
            if not isinstance(data, dict):
                return data
            else:
                return AttrDict(
                    {key: from_nested_dict(data[key])
                     for key in data}
                    )
        
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
        for key in self.keys():
            self[key] = from_nested_dict(self[key])


def _parse_args(config_parser, parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    
    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    if args.use_fixed_args:
        # The fix only works for up to two hierarchy levels (a.b, not a.b.c) but could be adapted
        args = use_fixed_args(args, cfg, remaining, checkout_missing_params=False)

    return args


def use_fixed_args(args, cfg, remaining, checkout_missing_params=False):
    args = AttrDict(jsonargparse.namespace_to_dict(args))
    remaining_set = set(remaining)
    for key, value in args.items():

        if isinstance(value, dict):
            # hierarchy level 2
            for keylow, _ in value.items():
                if f"--{key}.{keylow}" not in remaining_set:
                    args[key][keylow] = cfg[key][keylow]

        elif f"--{key}" not in remaining_set:
            # hierarchy level 1
            args[key] = cfg[key]

        # The fix only works if all the parameters are contained in the yaml file.
        # You can set 'checkout_missing_params' to True to print missing parameters
        if checkout_missing_params:
            print_missing_params_in_yaml(key, value, cfg)

    return args


def print_missing_params_in_yaml(key, value, cfg):
    if isinstance(value, dict):
        for keylow, _ in value.items():
            try:
                if keylow not in cfg[key]:
                    print(f"{key=},{keylow=}")
            except:
                pass