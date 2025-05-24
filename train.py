import collections
import glob
import numpy as np
import os
import sys
import time
import torch
import torch.optim as optim
import utils as U

from configuration import Configuration
from configuration import CONSTANTS as C
from data import AMASSBatch, LMDBDataset
from data_transforms import ExtractWindow, ToTensor
from evaluate import evaluate_test
from motion_metrics import MetricsEngine
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from models import MotionAttentionSPLModel
from losses import mse


def _log_loss_vals(loss_vals, writer, global_step, mode_prefix):
    for k, v in loss_vals.items():
        prefix = f'{k}/{mode_prefix}'
        writer.add_scalar(prefix, v, global_step)


def _evaluate(net, data_loader, metrics_engine):
    """
    Evaluate model on validation loader: compute average loss and metrics.
    """
    net.eval()
    loss_sum = 0.0
    n_samples = 0
    metrics_engine.reset()

    with torch.no_grad():
        for abatch in data_loader:
            batch_gpu = abatch.to_gpu()
            # extract seed sequence
            seq = batch_gpu.poses[:, :net.seed_len, :]
            predictions = net(seq)
            # ground-truth future frames
            targets = batch_gpu.poses[:, net.seed_len:]
            # compute loss
            loss = mse(predictions, targets)
            loss_sum += loss.item() * batch_gpu.batch_size
            # aggregate metrics
            metrics_engine.compute_and_aggregate(predictions, targets)
            n_samples += batch_gpu.batch_size

    avg_loss = loss_sum / n_samples if n_samples > 0 else 0.0
    return {'total_loss': avg_loss}


def main(config):
    # set seed
    if not hasattr(config, 'seed') or config.seed is None:
        config.seed = int(time.time())

    # transforms
    rng_extractor = np.random.RandomState(4313)
    window_size = config.seed_seq_len + config.target_seq_len
    train_transform = transforms.Compose([
        ExtractWindow(window_size, rng_extractor, mode='random'),
        ToTensor()
    ])
    valid_transform = transforms.Compose([ToTensor()])

    # datasets and loaders
    train_data = LMDBDataset(os.path.join(C.DATA_DIR, "training"), transform=train_transform,
                             filter_seq_len=window_size)
    valid_data = LMDBDataset(os.path.join(C.DATA_DIR, "validation"), transform=valid_transform)

    train_loader = DataLoader(train_data,
                              batch_size=config.bs_train,
                              shuffle=True,
                              num_workers=config.data_workers,
                              collate_fn=AMASSBatch.from_sample_list)
    valid_loader = DataLoader(valid_data,
                              batch_size=config.bs_eval,
                              shuffle=False,
                              num_workers=config.data_workers,
                              collate_fn=AMASSBatch.from_sample_list)

    # instantiate model
    net = MotionAttentionSPLModel(config).to(C.DEVICE)
    print(f'Model created with {U.count_parameters(net)} parameters')

    # metrics engine
    me = MetricsEngine(C.METRIC_TARGET_LENGTHS)

    # experiment setup
    experiment_id = int(time.time())
    experiment_name = 'motion_attention_spl'
    model_dir = U.create_model_dir(C.EXPERIMENT_DIR, experiment_id, experiment_name)
    code_files = glob.glob('./*.py', recursive=False)
    U.export_code(code_files, os.path.join(model_dir, 'code.zip'))
    config.to_json(os.path.join(model_dir, 'config.json'))
    with open(os.path.join(model_dir, 'cmd.txt'), 'w') as f:
        f.write(sys.argv[0] + ' ' + ' '.join(sys.argv[1:]))
    checkpoint_file = os.path.join(model_dir, 'model.pth')
    print(f'Saving checkpoints to {checkpoint_file}')

    writer = SummaryWriter(os.path.join(model_dir, 'logs'))
    optimizer = optim.SGD(net.parameters(), lr=config.lr)

    # training loop
    global_step = 0
    best_valid_loss = float('inf')
    for epoch in range(config.n_epochs):
        net.train()
        for i, abatch in enumerate(train_loader):
            start = time.time()
            optimizer.zero_grad()

            # forward: extract seed tensor and run the model
            batch_gpu = abatch.to_gpu()
            seq = batch_gpu.poses[:, :config.seed_seq_len, :]  # (B, seed_seq_len, input_dim)
            predictions = net(seq)

            # compute loss
            targets = batch_gpu.poses[:, config.seed_seq_len:, :]
            loss = mse(predictions, targets)
            loss_val = loss.item()

            # backward
            loss.backward()
            optimizer.step()

            # logging
            elapsed = time.time() - start
            loss_vals = {'total_loss': loss_val}
            _log_loss_vals(loss_vals, writer, global_step, 'train')
            writer.add_scalar('lr', config.lr, global_step)

            if global_step % config.print_every == 0:
                print(f'[TRAIN {i+1:05d} | {epoch+1:03d}] total_loss: {loss_val:.6f} elapsed: {elapsed:.3f}s')
                me.reset()
                me.compute_and_aggregate(predictions, targets)
                me.to_tensorboard_log(me.get_final_metrics(), writer, global_step, 'train')

            if global_step % config.eval_every == 0:
                # validation
                valid_losses = _evaluate(net, valid_loader, me)
                valid_metrics = me.get_final_metrics()
                print(f'[VALID] total_loss: {valid_losses["total_loss"]:.6f}')
                _log_loss_vals(valid_losses, writer, global_step, 'valid')

                # checkpoint
                if valid_losses['total_loss'] < best_valid_loss:
                    best_valid_loss = valid_losses['total_loss']
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'valid_loss': best_valid_loss,
                    }, checkpoint_file)
                net.train()

            global_step += 1

    # final test evaluation
    evaluate_test(experiment_id)


if __name__ == '__main__':
    main(Configuration.parse_cmd())
