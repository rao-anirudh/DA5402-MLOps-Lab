import json


def read_config(path_to_config="config.json"):

    """
    Reads a JSON configuration file and returns its contents as a Python dictionary
    :param path_to_config (str) - Path to the JSON configuration file
    :returns config (dict) - Python dictionary containing the configuration

    """

    with open(path_to_config, 'r') as f:
        config = json.load(f)
        f.close()

    return config
