import yaml

def load_config(config_path):

    config_file = open(config_path)
    parsed_config_file = yaml.load(config_file, Loader=yaml.FullLoader)

    return parsed_config_file


def configGUI(window_width, window_height, frame_rate):

    dict = {
        'window_width': window_width,
        'window_height': window_height,
        'frame_rate': frame_rate
    }

    with open('./config.yaml', 'w') as file:
        documents = yaml.dump(dict, file, sort_keys=True)