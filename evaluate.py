import argparse
import numpy as np
import os
import pandas as pd
import torch
import utils as U

from configuration import Configuration
from configuration import CONSTANTS as C
from data import AMASSBatch
from data import LMDBDataset
from data_transforms import ToTensor
from fk import SMPLForwardKinematics
from models import MotionAttentionSPLModel
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from visualize import Visualizer


def _export_results(eval_result, output_file):
    """
    Write predictions into a file that can be uploaded to the submission system.
    :param eval_result: A dictionary {sample_id => (prediction, seed)}
    :param output_file: Where to store the file.
    """
    def to_csv(fname, poses, ids, split=None):
        n_samples, seq_length, dof = poses.shape
        data_r = np.reshape(poses, [n_samples, seq_length * dof])
        cols = [f'dof{i}' for i in range(seq_length * dof)]

        if split is not None:
            data_r = np.concatenate([data_r, split[..., np.newaxis]], axis=-1)
            cols.append('split')

        df = pd.DataFrame(data_r, index=ids, columns=cols)
        df.index.name = 'Id'

        if not fname.endswith('.gz'):
            fname += '.gz'
        df.to_csv(fname, float_format='%.8f', compression='gzip')

    ids = []
    poses_list = []
    for sid, (pred, seed) in eval_result.items():
        ids.append(sid)
        poses_list.append(pred)
    to_csv(output_file, np.stack(poses_list), ids)


def load_model_weights(checkpoint_file, net, state_key='model_state_dict'):
    if not os.path.exists(checkpoint_file):
        raise ValueError(f"Could not find model checkpoint {checkpoint_file}.")
    ckpt = torch.load(checkpoint_file)[state_key]
    net.load_state_dict(ckpt)


def get_model_config(model_id):
    model_dir = U.get_model_dir(C.EXPERIMENT_DIR, model_id)
    model_config = Configuration.from_json(os.path.join(model_dir, 'config.json'))
    return model_config, model_dir


def load_model(model_id):
    model_config, model_dir = get_model_config(model_id)
    net = MotionAttentionSPLModel(model_config).to(C.DEVICE)
    print(f"Model created with {U.count_parameters(net)} trainable parameters")
    checkpoint_file = os.path.join(model_dir, 'model.pth')
    load_model_weights(checkpoint_file, net)
    print(f"Loaded weights from {checkpoint_file}")
    return net, model_config, model_dir


def evaluate_test(model_id, viz=False):
    net, model_config, model_dir = load_model(model_id)

    test_data = LMDBDataset(os.path.join(C.DATA_DIR, 'test'), transform=transforms.Compose([ToTensor()]))
    test_loader = DataLoader(
        test_data,
        batch_size=model_config.bs_eval,
        shuffle=False,
        num_workers=model_config.data_workers,
        collate_fn=AMASSBatch.from_sample_list
    )

    net.eval()
    results = {}
    with torch.no_grad():
        for abatch in test_loader:
            batch_gpu = abatch.to_gpu()
            # extract seed sequence and predict
            seq = batch_gpu.poses[:, :model_config.seed_seq_len, :]
            preds = net(seq)  # (B, target_seq_len, input_dim)
            seeds = batch_gpu.poses[:, :model_config.seed_seq_len, :]
            for b, sid in enumerate(batch_gpu.seq_ids):
                results[sid] = (preds[b].cpu().numpy(), seeds[b].cpu().numpy())

    fname = f'predictions_in{model_config.seed_seq_len}_out{model_config.target_seq_len}.csv'
    _export_results(results, os.path.join(model_dir, fname))

    if viz:
        fk_engine = SMPLForwardKinematics()
        visualizer = Visualizer(fk_engine)
        for sid, (pred, seed) in list(results.items())[:10]:
            visualizer.visualize(seed, pred, title=f'Sample ID: {sid}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', required=True, help='Which model to evaluate.')
    args = parser.parse_args()
    evaluate_test(args.model_id, viz=True)
