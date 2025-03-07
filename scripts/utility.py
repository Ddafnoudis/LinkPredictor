import yaml
from typing import Any, Dict

class Utility:
    def __init__(self):
        pass

    @staticmethod
    def parse_config(fname: str) -> Dict[str, Any]:
        """
        Parses YAML configuration file.

        :param fname: path to file
        :return: dictionary with configuration parameters
        """

        with open(fname, 'r') as f:
            contents = yaml.safe_load(f)

        return contents
