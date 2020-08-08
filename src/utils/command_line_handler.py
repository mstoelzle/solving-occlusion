import argparse


class CommandLineHandler:
    @staticmethod
    def handle():
        parser = argparse.ArgumentParser(description='ETHZ IDSC Frazzoli Path Learning Framework')
        parser.add_argument('config_path', metavar='config_path',
                            help='Specify the config file path within the /configs directory (for example '
                                 '"6_step_contrast.json")')
        # parser.add_argument('--example_flag', metavar='example_flag',
        #                     help='help for example_flag')

        args = parser.parse_args()
        config_path = args.config_path

        if config_path is None:
            raise Exception("No config path was specified")

        return args

    @staticmethod
    def bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
