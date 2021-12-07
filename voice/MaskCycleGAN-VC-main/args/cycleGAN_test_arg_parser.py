"""
Arguments for MaskCycleGAN-VC testing.
Inherits BaseArgParser.
"""

from args.base_arg_parser import BaseArgParser

class CycleGANTestArgParser(BaseArgParser):
    """
    Class which implements an argument parser for args used only in training MaskCycleGAN-VC.
    It inherits TrainArgParser.
    """

    def __init__(self):
        super(CycleGANTestArgParser, self).__init__()
        self.parser.add_argument('--sample_rate', type=int, default=22050, help='Sampling rate of mel-spectrograms.')
        self.parser.add_argument(
            '--speaker_A_id', type=str, default="VCC2SM5", help='Source speaker id (From VOC dataset).')
        self.parser.add_argument(
            '--speaker_B_id', type=str, default="SF3", help='Source speaker id (From VOC dataset).')
        self.parser.add_argument(
            '--preprocessed_data_dir', type=str, default="vcc2018_preprocessed/vcc2018_training", help='Directory containing preprocessed dataset files.')
        self.parser.add_argument(
            '--ckpt_dir', type=str, default="results/mask_cyclegan_vc_VCC2SF3_VCC2TF1/ckpts", help='Path to model ckpt.')
        self.parser.add_argument(
            '--model_name', type=str, choices=('generator_A2B', 'generator_B2A'), default='generator_B2A', help='Name of model to load.') #generator_A2B
