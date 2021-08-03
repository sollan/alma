import yaml


def load_config(config_path):

    config_file = open(config_path)
    parsed_config_file = yaml.load(config_file, Loader=yaml.FullLoader)

    return parsed_config_file
