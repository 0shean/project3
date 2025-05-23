"""
Data transformations to be applied to samples or batches before feeding the data to the model.

Copyright ETH Zurich, Manuel Kaufmann
"""
from data import AMASSSample
import numpy as np
from scipy.fftpack import dct, idct

class DCTTransform:
    def __init__(self, num_coeffs=20):
        self.num_coeffs = num_coeffs  # number of DCT coeffs to retain

    def __call__(self, sample):
        poses = sample.poses  # [T, 135] most likely
        T = poses.shape[0]

        if poses.ndim == 2 and poses.shape[1] == 135:
            poses = poses.reshape(T, 15, 9).reshape(T, 15, 3, 3)  # 💥 FIX HERE

        poses_flat = poses.reshape(T, 15, 9)

        dct_coeffs = np.zeros((15, 9, self.num_coeffs), dtype=np.float32)
        for j in range(15):
            for d in range(9):
                joint_signal = poses_flat[:, j, d]
                coeffs = dct(joint_signal, norm='ortho')
                dct_coeffs[j, d] = coeffs[:self.num_coeffs]

        sample.dct_input = dct_coeffs
        return sample


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
