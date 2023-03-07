import utils

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
    def _load(path:str):
        args = utils.load_json(path)
        for k,v in args.items():
            setattr(Config, k, v)
    
    input_dim = 100000
    hidden_dim = 100000
    output_dim = 100000
    num_layers = 100000
    num_epochs = 1000