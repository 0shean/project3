"""
Define common constants that are used globally.

Also define a configuration object whose parameters can be set via the command line or loaded from an existing
json file. Here you can add more configuration parameters that should be exposed via the command line. In the code,
you can access them via `config.your_parameter`. All parameters are automatically saved to disk in JSON format.

Copyright ETH Zurich, Manuel Kaufmann
"""
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
        parser = argparse.ArgumentParser()

        # … existing args …

        # Data.
        parser.add_argument('--seed_seq_len', type=int, default=120, help='Number of frames for the seed length.')
        parser.add_argument('--target_seq_len', type=int, default=24, help='How many frames to predict.')

        # Learning configurations.
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
        parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs.')
        parser.add_argument('--bs_train', type=int, default=16, help='Batch size for the training set.')
        parser.add_argument('--bs_eval', type=int, default=16, help='Batch size for valid/test set.')

        # Model (Motion-Attention + SPL) hyper-parameters
        parser.add_argument('--model_embed_dim', type=int, default=512,
                            help='Dimensionality of frame embeddings and attention output.')
        parser.add_argument('--model_use_gru', action='store_true',
                            help='If set, include a local GRU branch alongside motion-attention.')
        # Motion-Attention
        parser.add_argument('--ma_window_size', type=int, default=30,
                            help='DCT window length (in frames).')
        parser.add_argument('--ma_stride', type=int, default=15,
                            help='Hop size between DCT windows.')
        parser.add_argument('--ma_dct_size', type=int, default=20,
                            help='Number of DCT coefficients to keep per window.')
        parser.add_argument('--ma_num_heads', type=int, default=8,
                            help='Number of attention heads in motion-attention.')
        parser.add_argument('--ma_dropout', type=float, default=0.1,
                            help='Dropout in motion-attention module.')
        # Structured Prediction Layer
        parser.add_argument('--spl_hidden_size', type=int, default=256,
                            help='Hidden units in each per-joint MLP.')
        parser.add_argument('--spl_share_weights', action='store_true',
                            help='If set, all joints share the same MLP in SPL.')

        # Data‐specific constants that never change at runtime can live in Constants,
        # but if you prefer CLI control you can also expose them here:
        parser.add_argument('--input_dim', type=int, default=135,
                            help='Input dimensionality (n_joints * joint_dim).')
        parser.add_argument('--joint_dim', type=int, default=9,
                            help='Dimensionality of each joint rotation (e.g., 3×3 matrix flattened).')
        parser.add_argument('--parent_indices', nargs='+', type=int,
                            default=[-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11, 12, 1],
                            help='SMPL kinematic tree')
        # parent indices is awkward to pass by CLI; better to hard-code in a Constants or JSON:
        #   parent_indices = [-1, 0, 1, 2, 3, 1, 5, …]

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
