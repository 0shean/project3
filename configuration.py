import argparse
import json
import os
import pprint
import torch


class Constants(object):
    """
    This is a singleton.
    """
    class __Constants:
        def __init__(self):
            # Environment setup.
            self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.DTYPE = torch.float32
            self.DATA_DIR = os.environ['MP_DATA']
            self.EXPERIMENT_DIR = os.environ['MP_EXPERIMENTS']
            self.METRIC_TARGET_LENGTHS = [5, 10, 19, 24]  # @ 60 fps, in ms: 83.3, 166.7, 316.7, 400

    instance = None

    def __new__(cls, *args, **kwargs):
        if not Constants.instance:
            Constants.instance = Constants.__Constants()
        return Constants.instance

    def __getattr__(self, item):
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)


CONSTANTS = Constants()


class Configuration(object):
    """Configuration parameters exposed via the commandline."""

    def __init__(self, adict):
        self.__dict__.update(adict)

    def __str__(self):
        return pprint.pformat(vars(self), indent=4)

    @staticmethod
    def parse_cmd():
        parser = argparse.ArgumentParser(description="Configuration for GRU-TC motion forecasting")

        # General
        parser.add_argument('--data_workers', type=int, default=4,
                            help='Number of parallel threads for data loading.')
        parser.add_argument('--print_every', type=int, default=200,
                            help='Print stats to console every so many iters.')
        parser.add_argument('--eval_every', type=int, default=400,
                            help='Evaluate validation set every so many iters.')
        parser.add_argument('--tag', default='', help='A custom tag for this experiment.')
        parser.add_argument('--seed', type=int, default=None, help='Random number generator seed.')

        # Data
        parser.add_argument('--seed_seq_len', type=int, default=120,
                            help='Number of frames for the seed length.')
        parser.add_argument('--target_seq_len', type=int, default=24,
                            help='How many frames to predict.')

        # Batch sizes
        parser.add_argument('--bs_train', type=int, default=64, help='Batch size for the training set.')
        parser.add_argument('--bs_eval', type=int, default=128, help='Batch size for valid/test set.')

        # Learning configurations
        parser.add_argument('--lr', type=float, default=2.5e-4, help='Base learning rate.')
        parser.add_argument('--n_epochs', type=int, default=80, help='Number of epochs.')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 penalty on weights.')
        parser.add_argument('--lr_schedule', type=str, default='transformer',
                            choices=['transformer'], help='Which learning-rate schedule to use.')
        parser.add_argument('--lr_warmup_steps', type=int, default=4000,
                            help='Steps for Transformer LR warm-up.')

        # Model architecture
        parser.add_argument('--model', type=str, default='gru_tc', choices=['gru_tc'],
                            help='Which model to train (must be "gru_tc").')
        parser.add_argument('--d_model', type=int, default=256,
                            help='Transformer hidden dimension (e.g. 256).')
        parser.add_argument('--n_head', type=int, default=4,
                            help='Number of self-attention heads.')
        parser.add_argument('--n_layer', type=int, default=2,
                            help='Number of Transformer layers.')
        parser.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout for the Transformer context-conditioner.')

        # Loss weights
        parser.add_argument('--loss_geodesic', type=float, default=1.0,
                            help='Weight for geodesic loss.')
        parser.add_argument('--loss_vel', type=float, default=0.10,
                            help='Weight for velocity-diff loss.')
        parser.add_argument('--loss_bone', type=float, default=0.50,
                            help='Weight for bone-length loss.')
        parser.add_argument('--loss_limit', type=float, default=0.50,
                            help='Weight for joint-limit loss.')
        parser.add_argument('--loss_pskld', type=float, default=0.20,
                            help='Weight for power-spectrum KLD self-distillation.')

        # Curriculum noise
        parser.add_argument('--curriculum_start_std', type=float, default=0.05,
                            help='Initial sigma for curriculum noise (radians).')
        parser.add_argument('--curriculum_end_std', type=float, default=0.20,
                            help='Final sigma for curriculum noise.')
        parser.add_argument('--curriculum_steps', type=int, default=40000,
                            help='Number of training steps to ramp noise.')

        # Diffusion refinement
        parser.add_argument('--diffusion_steps', type=int, default=6,
                            help='Number of diffusion steps at inference.')
        parser.add_argument('--diffusion_warmup_epochs', type=int, default=10,
                            help='Epochs to wait before enabling diffusion.')

        # Additional
        parser.add_argument('--input_dim', type=int, default=135,
                            help='Dimensionality of input pose (6-D times joints or matrix).')

        config = parser.parse_args()
        return Configuration(vars(config))

    @staticmethod
    def from_json(json_path):
        """Load configurations from a JSON file."""
        with open(json_path, 'r') as f:
            config = json.load(f)
            return Configuration(config)

    def to_json(self, json_path):
        """Dump configurations to a JSON file."""
        with open(json_path, 'w') as f:
            s = json.dumps(vars(self), indent=2, sort_keys=True)
            f.write(s)
