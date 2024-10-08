import argparse
from omegaconf import OmegaConf


class BaseConfig:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument(
            "--task_config",
            type=str,
            default="configs/qm9_pio.yaml",
        )
        self.initialized = True
        return parser

    def parse(self, verbose=True):
        if not self.initialized:
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()

        task_config = OmegaConf.load(opt.task_config)

        if verbose:
            print("")
            print("----------------- Task Config ---------------\n")
            print(OmegaConf.to_yaml(task_config))
            print("------------------- End ----------------\n")
            print("")

        return task_config
