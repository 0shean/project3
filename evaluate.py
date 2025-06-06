"""
Evaluate a GRU‑TC model on the MP‑Project3 **test** split.

Changes vs. original ETH template
---------------------------------
* Works with the new `GRUTCMotionForecast` that outputs a **tensor** `(B,T,J,6)`.
* Rebuilds the model **configuration** from the checkpoint (`model.pth`) when the
  old `config.json` file isn’t present (our new trainer didn’t write it).
* Converts seed 9‑D → 6‑D before inference and converts predictions 6‑D → 9‑D
  before writing the CSV.
"""
from __future__ import annotations
import argparse, os, numpy as np, pandas as pd, torch
import utils as U

from configuration import Configuration, CONSTANTS as C
from data import AMASSBatch, LMDBDataset
from data_transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models import GRUTCMotionForecast, rot6d_to_matrix, matrix_to_rot6d

# ────────────────────────────────────────────────────────────────────

def _export_results(eval_result, output_file):
    """Write predictions into a gz‑compressed CSV."""
    def to_csv(fname, poses, ids):
        n_samples, seq_len, dof = poses.shape
        data_r = poses.reshape(n_samples, seq_len * dof)
        cols = [f"dof{i}" for i in range(seq_len * dof)]
        df = pd.DataFrame(data_r, index=ids, columns=cols)
        df.index.name = "Id"
        if not fname.endswith(".gz"):
            fname += ".gz"
        df.to_csv(fname, float_format="%.8f", compression="gzip")

    ids, preds = zip(*[(k, v) for k, v in eval_result.items()])
    to_csv(output_file, np.stack(preds), list(ids))

# ────────────────────────────────────────────────────────────────────

def load_checkpoint(model_id: str):
    """Return (cfg, ckpt_path, model_dir)."""
    model_dir = U.get_model_dir(C.EXPERIMENT_DIR, model_id)
    ckpt_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} not found")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Rebuild Configuration either from ckpt["cfg"] (new) or config.json (old)
    if "cfg" in ckpt:
        cfg = Configuration(ckpt["cfg"])
    else:
        cfg = Configuration.from_json(os.path.join(model_dir, "config.json"))
    return cfg, ckpt_path, model_dir

# ────────────────────────────────────────────────────────────────────

def to_seed6d(seed9: torch.Tensor, cfg) -> torch.Tensor:
    """(B,T,J*9) → (B,T,J,6) using matrix_to_rot6d."""
    B, T, D = seed9.shape
    J = D // 9
    mat = seed9.reshape(B, T, J, 3, 3)
    seed6 = matrix_to_rot6d(mat.reshape(-1, 3, 3)).reshape(B, T, J, 6)
    return seed6

# ────────────────────────────────────────────────────────────────────

def evaluate_test(model_id: str):
    cfg, ckpt_path, model_dir = load_checkpoint(model_id)

    # Build model & load weights
    net = GRUTCMotionForecast(cfg).to(C.DEVICE)
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model_state"])
    net.eval()

    # DataLoader over test set (only seeds are stored)
    test_tf = transforms.Compose([ToTensor()])
    test_ds = LMDBDataset(os.path.join(C.DATA_DIR, "test"), transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=cfg.bs_eval, shuffle=False,
                              num_workers=cfg.data_workers, collate_fn=AMASSBatch.from_sample_list)

    results = {}
    with torch.no_grad():
        for abatch in test_loader:
            b_gpu = abatch.to_gpu()
            seed6 = to_seed6d(b_gpu.poses, cfg)           # test set only has the seed
            pred6 = net(seed6)                            # (B,T,J,6)

            # Convert predictions to 9‑D flattened
            B, T, J, _ = pred6.shape
            pred9 = rot6d_to_matrix(pred6.reshape(-1, 6)).reshape(B, T, J * 9)

            for i in range(B):
                results[b_gpu.seq_ids[i]] = pred9[i].detach().cpu().numpy()

    out_file = os.path.join(model_dir, f"predictions_in{cfg.seed_seq_len}_out{cfg.target_seq_len}.csv")
    _export_results(results, out_file)
    print(f"Predictions written to {out_file}")

# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True, help="Which experiment ID to evaluate")
    args = parser.parse_args()
    evaluate_test(args.model_id)
