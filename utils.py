"""
Some utility functions.

Copyright ETH Zurich, Manuel Kaufmann
"""
import glob
import os
import zipfile
import torch
import math


def dct(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """
    Compute the 1D Discrete Cosine Transform (Type II) along the last dimension of x.
    Args:
        x: Tensor of shape (..., N)
        norm: 'ortho' for orthonormal normalization, otherwise no normalization
    Returns:
        Tensor of same shape as x containing DCT coefficients.
    """
    N = x.shape[-1]
    # create basis matrix
    n = torch.arange(N, device=x.device).unsqueeze(1)  # (N, 1)
    k = torch.arange(N, device=x.device).unsqueeze(0)  # (1, N)
    basis = torch.cos(math.pi / (2 * N) * (2 * n + 1) * k)  # (N, N)
    # apply transform
    X = torch.matmul(x, basis)  # (..., N) @ (N, N) -> (..., N)
    if norm == 'ortho':
        X[..., 0] = X[..., 0] / math.sqrt(N)
        X[..., 1:] = X[..., 1:] * math.sqrt(2.0 / N)
    return X


def idct(X: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """
    Compute the 1D Inverse Discrete Cosine Transform (Type II inverse) along the last dimension of X.
    Args:
        X: Tensor of shape (..., N) with DCT coefficients
        norm: 'ortho' for orthonormal normalization, otherwise no normalization
    Returns:
        Tensor of same shape as X representing the reconstructed signal.
    """
    N = X.shape[-1]
    # create basis matrix
    n = torch.arange(N, device=X.device).unsqueeze(1)  # (N, 1)
    k = torch.arange(N, device=X.device).unsqueeze(0)  # (1, N)
    basis = torch.cos(math.pi / (2 * N) * (2 * n + 1) * k)  # (N, N)
    # undo normalization
    Y = X.clone()
    if norm == 'ortho':
        Y[..., 0] = Y[..., 0] * math.sqrt(N)
        Y[..., 1:] = Y[..., 1:] / math.sqrt(2.0 / N)
    # inverse transform
    x = torch.matmul(Y, basis) * (2.0 / N)
    return x



def create_model_dir(experiment_main_dir, experiment_id, model_summary):
    """
    Create a new model directory.
    :param experiment_main_dir: Where all experiments are stored.
    :param experiment_id: The ID of this experiment.
    :param model_summary: A summary string of the model.
    :return: A directory where we can store model logs. Raises an exception if the model directory already exists.
    """
    model_name = "{}-{}".format(experiment_id, model_summary)
    model_dir = os.path.join(experiment_main_dir, model_name)
    if os.path.exists(model_dir):
        raise ValueError("Model directory already exists {}".format(model_dir))
    os.makedirs(model_dir)
    return model_dir


def get_model_dir(experiment_dir, model_id):
    """Return the directory in `experiment_dir` that contains the given `model_id` string."""
    model_dir = glob.glob(os.path.join(experiment_dir, str(model_id) + "-*"), recursive=False)
    return None if len(model_dir) == 0 else model_dir[0]


def export_code(file_list, output_file):
    """Stores files in a zip."""
    if not output_file.endswith('.zip'):
        output_file += '.zip'
    ofile = output_file
    counter = 0
    while os.path.exists(ofile):
        counter += 1
        ofile = output_file.replace('.zip', '_{}.zip'.format(counter))
    zipf = zipfile.ZipFile(ofile, mode="w", compression=zipfile.ZIP_DEFLATED)
    for f in file_list:
        zipf.write(f)
    zipf.close()


def count_parameters(net):
    """Count number of trainable parameters in `net`."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
