from typing import Union
import argparse
import json

def load_json(path):
    with open(path,'r') as f:
        res = json.load(f)
    return res


def save_json(obj, path:str):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=4)


class Config:
    @staticmethod
    def _save(path:str):
        args = Config._to_dict()
        save_json(args, path)


    @staticmethod
    def _display(n:int=60):
        print("*"*n)
        for k in sorted(dir(Config)):
            if not k.startswith("_"):
                print("{:<20}\t{}".format(k, getattr(Config, k)))
        print("*"*n)


    @staticmethod
    def _to_dict():
        args = {}
        for k in sorted(dir(Config)):
            if not k.startswith("_"):
                args[k] = getattr(Config, k)
        return args
    

    @staticmethod
    def _load(args:Union[str, dict, argparse.Namespace]):
        """
        @brief load config
        str: path to json
        dict
        argparse.Namespace
        """
        if isinstance(args, str):
            args = load_json(args)
        elif isinstance(args, dict):
            pass
        elif isinstance(args, argparse.Namespace):
            args = dict(vars(args))
        else:
            raise NotImplementedError(f"type of args is {type(args)}, not implemented")
        for k,v in args.items():
            setattr(Config, k, v)
    

    input_dim = 32
    hidden_dim = 64
    output_dim = 16
    num_layers = 4
    num_epochs = 100



if __name__ == "__main__":
    args = {
        "name": "Ruizhe Zhong",
        "id": "519021910025",
    }

    Config._load(args)

    Config._display()
    args = Config._to_dict()
    for k,v in args.items():
        print(k,v)
    
    config = Config()
    config._display()
    Config._save("args.json")