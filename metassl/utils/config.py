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
    
    return args