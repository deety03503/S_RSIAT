import json
import os
import argparse
from trainer import RSIAT_train
from utils.sdrsiat_config import add_sdrsiat_cli_args, pop_sdrsiat_cli, apply_sdrsiat_overrides

def main():
    args = parse_arguments()
    sdrsiat_cli = pop_sdrsiat_cli(args)  # SD-RSIAT flags: CLI (if given) overrides JSON
    config = load_json(args.config)
    merged_config = merge_configs(args, config)
    apply_sdrsiat_overrides(merged_config, sdrsiat_cli)
    RSIAT_train(merged_config)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    """
    Please specify the corresponding JSON file!!!!
    """
    parser.add_argument('--config', type=str, default='./exps/adapter_imageneta.json',
                        help='Json file of settings.')
    add_sdrsiat_cli_args(parser)
    return parser.parse_args()

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

def merge_configs(args, config):
    merged_config = vars(args)  
    merged_config.update(config)  
    return merged_config

if __name__ == '__main__':
    main()