import utils
from typing import Union
import argparse

class Config:
    @staticmethod
    def _save(path:str):
        args = {}
        for k in sorted(dir(Config)):
            if not k.startswith("_"):
                args[k] = getattr(Config, k)
        utils.save_json(args, path)

    @staticmethod
    def _display(n:int=60):
        print("*"*n)
        for k in sorted(dir(Config)):
            if not k.startswith("_"):
                print("{:<20}\t{}".format(k, getattr(Config, k)))
        print("*"*n)

    @staticmethod
    def _load(args:Union[str, dict, argparse.Namespace]):
        if isinstance(args, str):
            args = utils.load_json(args)
        elif isinstance(args, dict):
            pass
        elif isinstance(args, argparse.Namespace):
            args = dict(vars(args))
        else:
            raise NotImplementedError(f"type of args is {type(args)}, not implemented")
        for k,v in args.items():
            setattr(Config, k, v)
    
    input_dim = 100000
    hidden_dim = 100000
    output_dim = 100000
    num_layers = 100000
    num_epochs = 1000