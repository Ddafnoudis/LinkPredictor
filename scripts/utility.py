import yaml
from pathlib import Path
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

        # create output path if it exists
        if not ('output_dir' in contents.keys()):
            raise ValueError('Not output directory specified! Add variable "output_dir" to configuration file.')

        #
        contents['output_dir'] = Path(contents['output_dir'])
        if not contents['output_dir'].exists():
            contents['output_dir'].mkdir(exist_ok=True, parents=True)

        return contents
