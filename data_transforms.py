"""
Data transformations to be applied to samples or batches before feeding the data to the model.

Copyright ETH Zurich, Manuel Kaufmann
"""
from data import AMASSSample
import numpy as np
from scipy.fftpack import dct, idct

import numpy as np

class DCTTransform:
    def __init__(self, num_coeffs=40, window_len=50, step_size=10):
        self.num_coeffs = num_coeffs
        self.window_len = window_len
        self.step_size = step_size

    def __call__(self, sample):
        poses = sample.poses
        T = poses.shape[0]

        if poses.ndim == 2 and poses.shape[1] == 135:
            poses = poses.reshape(T, 15, 9).reshape(T, 15, 3, 3)

        poses_flat = poses.reshape(T, 15, 9)

        # === 1. Extract query (last 50 frames)
        query = poses_flat[-self.window_len:]  # [50, 15, 9]
        dct_input = self._to_dct(query)  # [15, 9, K]
        sample.dct_input = dct_input

        # === 2. Extract motion bank from earlier seed
        history = []
        for start in range(0, T - self.window_len - 24 + 1, self.step_size):
            window = poses_flat[start:start + self.window_len]
            history.append(self._to_dct(window))

        if history:
            sample.dct_history = np.stack(history)  # [N, 15, 9, K]
        else:
            sample.dct_history = np.zeros((1, 15, 9, self.num_coeffs), dtype=np.float32)

        return sample

    def _to_dct(self, seq):
        # seq: [T, 15, 9]
        T = seq.shape[0]
        coeffs = np.zeros((15, 9, self.num_coeffs), dtype=np.float32)
        for j in range(15):
            for d in range(9):
                signal = seq[:, j, d]
                dct_coeff = dct(signal, norm='ortho')[:self.num_coeffs]
                coeffs[j, d] = dct_coeff
        return coeffs



class ToTensor(object):
    """Convert numpy arrays inside samples to PyTorch tensors."""

    def __call__(self, sample: AMASSSample):
        sample.to_tensor()
        return sample


class ExtractWindow(object):
    """
    Extract a window of a fixed size. If the sequence is shorter than the desired window size it will return the
    entire sequence without any padding.
    """

    def __init__(self, window_size, rng=None, mode='random'):
        assert mode in ['random', 'beginning', 'middle']
        if mode == 'random':
            assert rng is not None
        self.window_size = window_size
        self.rng = rng
        self.mode = mode
        self.padding_value = 0.0

    def __call__(self, sample: AMASSSample):
        if sample.n_frames > self.window_size:
            if self.mode == 'beginning':
                sf, ef = 0, self.window_size
            elif self.mode == 'middle':
                mid = sample.n_frames // 2
                sf = mid - self.window_size // 2
                ef = sf + self.window_size
            elif self.mode == 'random':
                sf = self.rng.randint(0, sample.n_frames - self.window_size + 1)
                ef = sf + self.window_size
            else:
                raise ValueError("Mode '{}' for window extraction unknown.".format(self.mode))
            return sample.extract_window(sf, ef)
        else:
            return sample
