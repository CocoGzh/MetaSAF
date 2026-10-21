import os
import time
import random
from math import sqrt
import shap
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm
from sklearn.model_selection import KFold, train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func


# =========================
# Reproducibility
# =========================
def seed_everything(seed: int = 42, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =========================
# Flash Transformer Blocks
# =========================
class FlashTransformerBlockVarlen(nn.Module):
    def __init__(self, d_model=512, n_heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def _compute_attention_maps(self, qkv, cu_seqlens, detach_attn=True):
        """
        返回:
          attn_maps: list length = batch_size
            attn_maps[b]: (H, L_b, L_b)
        注意:
          L_b 是该 sample 中 valid bacteria 的数量。
        """
        q = qkv[:, 0]  # (N, H, Hd)
        k = qkv[:, 1]  # (N, H, Hd)

        if detach_attn:
            q = q.detach()
            k = k.detach()

        q = q.float()
        k = k.float()

        scale = self.head_dim ** -0.5
        bs = cu_seqlens.numel() - 1

        attn_maps = []
        for b in range(bs):
            start = int(cu_seqlens[b].item())
            end = int(cu_seqlens[b + 1].item())

            qb = q[start:end].transpose(0, 1)  # (H, L, Hd)
            kb = k[start:end].transpose(0, 1)  # (H, L, Hd)

            scores = torch.matmul(qb, kb.transpose(-2, -1)) * scale  # (H, L, L)
            attn = torch.softmax(scores, dim=-1)                     # (H, L, L)

            attn_maps.append(attn)

        return attn_maps

    def forward(self, x, cu_seqlens, max_seqlen, return_attn=False, detach_attn=True):
        residual = x
        x = self.norm1(x)

        qkv = self.qkv(x).view(-1, 3, self.n_heads, self.head_dim)

        attn_maps = None
        if return_attn:
            attn_maps = self._compute_attention_maps(
                qkv,
                cu_seqlens=cu_seqlens,
                detach_attn=detach_attn,
            )

        attn_out = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False,
        )  # (N, H, Hd)

        attn_out = attn_out.reshape(-1, self.d_model)
        x = self.out_proj(attn_out)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual

        if return_attn:
            return x, attn_maps
        return x


class FlashBactEncoder(nn.Module):
    def __init__(self, d_model=512, n_heads=8, mlp_ratio=4, dropout=0.1, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            FlashTransformerBlockVarlen(
                d_model=d_model,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

    def forward(self, x, cu_seqlens, max_seqlen, return_attn=False, detach_attn=True):
        all_attn_maps = []

        for layer in self.layers:
            if return_attn:
                x, attn_maps = layer(
                    x,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    return_attn=True,
                    detach_attn=detach_attn,
                )
                all_attn_maps.append(attn_maps)
            else:
                x = layer(
                    x,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )

        if return_attn:
            return x, all_attn_maps

        return x


# =========================
# Dataset / Loader
# =========================
class MeanPooledPersonDataset(Dataset):
    def __init__(
        self,
        df,
        X_count,
        bact_mean_all,
        mask_all,
        n_reads_all,
        label_col="age",
        id_col=None,
        filter_missing_label=True,
        row_idx_col="row_idx",
        label_dtype=np.float32,
        log1p_count=True,
    ):
        df = df.reset_index(drop=True).copy()

        if row_idx_col is None:
            base_indices = np.arange(len(df), dtype=np.int64)
        else:
            if row_idx_col not in df.columns:
                raise ValueError(f"row_idx_col='{row_idx_col}' not found in df.columns")
            base_indices = df[row_idx_col].to_numpy(np.int64)

        if filter_missing_label:
            valid_mask = df[label_col].notna().to_numpy()
            kept = int(valid_mask.sum())
            total = int(len(valid_mask))
            if kept == 0:
                raise ValueError(f"No valid samples found for label '{label_col}'")
            df = df.loc[valid_mask].reset_index(drop=True)
            base_indices = base_indices[valid_mask]
            print(f"[Dataset] label={label_col}: keep {kept}/{total} samples")
        else:
            print(f"[Dataset] label={label_col}: keep {len(df)}/{len(df)} samples")

        self.df = df
        self.label_col = label_col
        self.id_col = id_col
        self.base_indices = np.asarray(base_indices, dtype=np.int64)
        self.log1p_count = log1p_count

        # 这里不要切片复制，只保留引用
        self.X_count_all = X_count
        self.bact_mean_all = bact_mean_all
        self.mask_all = mask_all
        self.n_reads_all = n_reads_all

        self.labels = self.df[label_col].to_numpy(label_dtype)

        if id_col is not None:
            self.ids = self.df[id_col].tolist()
        else:
            self.ids = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        real_idx = self.base_indices[idx]

        bact_emb = self.bact_mean_all[real_idx].astype(np.float32, copy=False)
        mask = self.mask_all[real_idx].astype(np.bool_, copy=False)
        n_reads = self.n_reads_all[real_idx].astype(np.int64, copy=False)
        y = self.labels[idx]

        x_count = self.X_count_all[real_idx]
        if hasattr(x_count, "toarray"):
            x_count = x_count.toarray().squeeze(0)
        else:
            x_count = np.asarray(x_count)

        x_count = x_count.astype(np.float32, copy=False)
        if self.log1p_count:
            x_count = np.log1p(x_count)

        sample = {
            "bact_emb": torch.from_numpy(np.ascontiguousarray(bact_emb)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "n_reads": torch.from_numpy(np.ascontiguousarray(n_reads)),
            "count": torch.from_numpy(np.ascontiguousarray(x_count)),
            "y": torch.tensor(y, dtype=torch.float32),
        }

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


def collate_meanpooled_batch(batch):
    bact_emb = torch.stack([s["bact_emb"] for s in batch], dim=0)  # (bs, B, D)
    mask = torch.stack([s["mask"] for s in batch], dim=0)          # (bs, B)
    n_reads = torch.stack([s["n_reads"] for s in batch], dim=0)    # (bs, B)
    count = torch.stack([s["count"] for s in batch], dim=0)
    y = torch.stack([s["y"] for s in batch], dim=0).view(-1)

    out = {
        "bact_emb": bact_emb,
        "mask": mask,
        "n_reads": n_reads,
        "count": count,
        "y": y,
    }

    if "id" in batch[0]:
        out["id"] = [s["id"] for s in batch]

    return out


def make_loader_meanpooled(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_meanpooled_batch,
        # persistent_workers=(num_workers > 0),
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=g,
    )


# =========================
# Model
# =========================
class BacteriaMeanRegressor(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_bacteria: int = 1000,
        count_dim: int = 2000,
        count_hidden1: int = 128,
        count_hidden2: int = 16,
        bact_tf_layers: int = 1,
        bact_tf_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        sub_dim: int = 4,
    ):
        super().__init__()
        self.n_bacteria = n_bacteria
        self.d_model = d_model

        self.bact_flash_encoder = FlashBactEncoder(
            d_model=d_model,
            n_heads=bact_tf_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            n_layers=bact_tf_layers,
        )

        self.fc_squeeze = nn.Linear(d_model, sub_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(n_bacteria * sub_dim),
            nn.Linear(n_bacteria * sub_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(128),
            nn.Linear(128, 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(8, 1),
        )

        self.count_head = nn.Sequential(
            nn.LayerNorm(count_dim),
            nn.Linear(count_dim, count_hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(count_hidden1),
            nn.Linear(count_hidden1, count_hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(count_hidden2, 1),
        )

        self.alpha = nn.Parameter(torch.tensor(0.0))

    def _attn_maps_to_padded(self, attn_maps_local, valid, B):
        """
        attn_maps_local:
          list length = n_layers
            each layer: list length = bs
              tensor: (H, L_b, L_b)

        返回:
          list length = n_layers
            each tensor: (bs, H, B, B)
        """
        bs = valid.size(0)
        device = valid.device
        padded_layers = []

        for layer_maps in attn_maps_local:
            H = layer_maps[0].size(0) if len(layer_maps) > 0 else self.bact_flash_encoder.layers[0].n_heads
            padded = torch.zeros((bs, H, B, B), device=device, dtype=torch.float32)

            for b, attn in enumerate(layer_maps):
                idx = valid[b].nonzero(as_tuple=True)[0]
                L = idx.numel()

                if L == 0:
                    continue

                rr, cc = torch.meshgrid(idx, idx, indexing="ij")
                padded[b, :, rr, cc] = attn

            padded_layers.append(padded)

        return padded_layers

    def forward(
            self,
            bact_emb: torch.Tensor,
            bact_mask: torch.Tensor,
            count: torch.Tensor,
            return_attn: bool = False,
            attn_format: str = "padded",  # "padded" or "local"
            detach_attn: bool = True,
    ):
        """
        bact_emb:  (bs, B, D)
        bact_mask: (bs, B)
        count:     (bs, count_dim)

        return_attn=False:
          return pred_final, pred_seq, pred_count

        return_attn=True:
          return pred_final, pred_seq, pred_count, attn_maps

        attn_format="local":
          attn_maps[layer][batch]: (H, L_valid, L_valid)

        attn_format="padded":
          attn_maps[layer]: (bs, H, B, B)
          其中 invalid bacteria 的行/列为 0。
        """
        if attn_format not in {"padded", "local"}:
            raise ValueError(f"attn_format must be 'padded' or 'local', got {attn_format}")

        device = bact_emb.device
        bact_mask = bact_mask.to(device=device)

        if bact_mask.dtype != torch.bool:
            bact_mask = bact_mask != 0

        bs, B, D = bact_emb.shape

        if B != self.n_bacteria:
            raise ValueError(f"Expected B={self.n_bacteria}, got {B}")

        if D != self.d_model:
            raise ValueError(f"Expected D={self.d_model}, got {D}")

        valid = bact_mask
        n_valid_per = valid.sum(dim=1).to(dtype=torch.int32)

        total_valid = int(n_valid_per.sum().item())
        attn_maps = None

        if total_valid == 0:
            pred_seq = torch.zeros((bs,), device=device, dtype=torch.float32)

            if return_attn:
                H = self.bact_flash_encoder.layers[0].n_heads
                n_layers = len(self.bact_flash_encoder.layers)

                if attn_format == "padded":
                    attn_maps = [
                        torch.zeros((bs, H, B, B), device=device, dtype=torch.float32)
                        for _ in range(n_layers)
                    ]
                else:
                    attn_maps = [
                        [
                            torch.empty((H, 0, 0), device=device, dtype=torch.float32)
                            for _ in range(bs)
                        ]
                        for _ in range(n_layers)
                    ]

        else:
            bact_tokens = bact_emb[valid]  # (N_valid_bacteria, D)

            cu_seqlens = torch.zeros((bs + 1,), device=device, dtype=torch.int32)
            cu_seqlens[1:] = torch.cumsum(n_valid_per, dim=0)
            max_seqlen = int(n_valid_per.max().item())

            if return_attn:
                y, attn_maps_local = self.bact_flash_encoder(
                    bact_tokens,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    return_attn=True,
                    detach_attn=detach_attn,
                )

                if attn_format == "padded":
                    attn_maps = self._attn_maps_to_padded(
                        attn_maps_local,
                        valid=valid,
                        B=B,
                    )
                else:
                    attn_maps = attn_maps_local

            else:
                y = self.bact_flash_encoder(
                    bact_tokens,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )

            h = torch.zeros((bs, self.n_bacteria, self.d_model), device=device, dtype=y.dtype)
            h[valid] = y

            h2 = self.fc_squeeze(h)
            person_emb = h2.reshape(bs, -1)
            pred_seq = self.head(person_emb).squeeze(-1)

        count = count.to(device=device, dtype=torch.float32)
        pred_count = self.count_head(count).squeeze(-1)

        w = torch.sigmoid(self.alpha)
        pred_final = w * pred_seq + (1 - w) * pred_count

        if return_attn:
            if attn_format == "padded":
                attn_maps = [
                    layer_attn.mean(dim=1)  # (bs, H, B, B) -> (bs, B, B)
                    for layer_attn in attn_maps
                ]
            else:
                attn_maps = [
                    [
                        attn.mean(dim=0)  # (H, L, L) -> (L, L)
                        for attn in layer_maps
                    ]
                    for layer_maps in attn_maps
                ]
            return pred_final, pred_seq, pred_count, attn_maps

        return pred_final, pred_seq, pred_count


class CountOnlyWrapper(nn.Module):
    def __init__(self, base_model, mode="count"):
        """
        mode:
          "count"          -> 解释 pred_count
          "weighted_count" -> 解释 count 分支对 pred_final 的加权贡献: (1-w)*pred_count
        """
        super().__init__()
        self.base_model = base_model
        self.mode = mode

    def forward(self, count):
        count = count.float()
        pred_count = self.base_model.count_head(count).squeeze(-1)

        if self.mode == "weighted_count":
            w = torch.sigmoid(self.base_model.alpha)
            pred_count = (1.0 - w) * pred_count

        return pred_count.unsqueeze(-1)


def sample_indices(n, k, seed):
    rng = np.random.default_rng(seed)
    k = min(k, n)
    return rng.choice(n, size=k, replace=False)


def collect_count_tensor(ds, indices, device):
    xs = []
    for i in indices:
        item = ds[int(i)]
        x = item["count"]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        xs.append(x.float())
    return torch.stack(xs, dim=0).to(device)


# =========================
# Train / Eval
# =========================
@torch.no_grad()
def eval_one_epoch_meanpooled(model, loader, device, use_amp=True, amp_dtype=torch.float16, return_attn=False):
    model.eval()

    total_loss = 0.0
    n = 0
    preds_all, preds_all_count, preds_all_seq, targets_all = [], [], [], []
    atte_map_all = []
    for batch in loader:
        bact_emb = batch["bact_emb"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        count = batch["count"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).float()

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(use_amp and device.startswith("cuda"))):
            if return_attn:
                pred, pred_seq, pred_count, atte_map = model(bact_emb, mask, count, return_attn=return_attn, attn_format="padded")
                atte_map_all.extend(atte_map[-1].detach().cpu().numpy())  # batchsize, heads, bacteria, bacteria
            else:
                pred, pred_seq, pred_count = model(bact_emb, mask, count, return_attn=return_attn, attn_format="padded")


        loss_seq = F.mse_loss(pred_seq, y, reduction="sum")
        loss_count = F.mse_loss(pred_count, y, reduction="sum")
        loss = F.mse_loss(pred, y, reduction="sum") + loss_seq + loss_count

        total_loss += loss.item()
        n += y.numel()

        preds_all.extend(pred.detach().cpu().tolist())
        preds_all_count.extend(pred_count.detach().cpu().tolist())
        preds_all_seq.extend(pred_seq.detach().cpu().tolist())
        targets_all.extend(y.detach().cpu().tolist())

    mse = total_loss / max(1, n)
    rmse = sqrt(mse)
    mae = float(np.mean([abs(p - t) for p, t in zip(preds_all, targets_all)])) if n else 0.0

    if n >= 2 and np.std(preds_all) > 0 and np.std(targets_all) > 0:
        pcc = float(np.corrcoef(preds_all, targets_all)[0, 1])
    else:
        pcc = float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "pcc": pcc,
        "n": n,
        "preds_all": preds_all,
        "pred_count": preds_all_count,
        "pred_seq": preds_all_seq,
        "targets_all": targets_all,
        "atte_map_all": atte_map_all,
        # "atte_map": atte_map, # TODO
    }


def train_one_epoch_meanpooled(
    model,
    loader,
    optimizer,
    scaler,
    device,
    grad_clip=1.0,
    epoch=1,
    accumulate_steps=8,
    use_amp=True,
    amp_dtype=torch.float16,
):
    model.train()

    total_loss = 0.0
    n = 0
    t0 = time.time()
    preds_all = []
    targets_all = []
    preds_all_count = []
    preds_all_seq = []
    optimizer.zero_grad(set_to_none=True)

    step_idx = 0
    pbar = tqdm(loader, desc=f"Train E{epoch}", leave=False)

    for step_idx, batch in enumerate(pbar, start=1):
        bact_emb = batch["bact_emb"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        count = batch["count"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).float()

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(use_amp and device.startswith("cuda"))):
            pred, pred_seq, pred_count = model(bact_emb, mask, count)

            loss_seq = F.mse_loss(pred_seq, y, reduction="mean") / accumulate_steps
            loss_count = F.mse_loss(pred_count, y, reduction="mean") / accumulate_steps
            loss = loss_seq + loss_count + F.mse_loss(pred, y, reduction="mean") / accumulate_steps
            # print(len(pred), len(pred_count), len(pred_seq), len(y))
            preds_all.extend(pred.detach().cpu().tolist())
            preds_all_count.extend(pred_count.detach().cpu().tolist())
            preds_all_seq.extend(pred_seq.detach().cpu().tolist())
            targets_all.extend(y.detach().cpu().tolist())

        scaler.scale(loss).backward()

        bs = y.numel()
        total_loss += loss.item() * accumulate_steps * bs
        n += bs

        # preds_all.extend(pred.detach().cpu().tolist())
        # targets_all.extend(y.detach().cpu().tolist())

        if step_idx % accumulate_steps == 0:
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        pbar.set_postfix(
            loss=f"{(total_loss / max(1, n)):.4f}",
            acc=f"{step_idx % accumulate_steps}/{accumulate_steps}",
        )

    if step_idx > 0 and (step_idx % accumulate_steps) != 0:
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if n >= 2 and np.std(preds_all) > 0 and np.std(targets_all) > 0:
        pcc = float(np.corrcoef(preds_all, targets_all)[0, 1])
    else:
        pcc = float("nan")

    epoch_time = time.time() - t0
    return {
        "mse": total_loss / max(1, n),
        "pcc": pcc,
        "n": n,
        "sec": epoch_time,

        "preds_all": preds_all,
        "pred_count": preds_all_count,
        "pred_seq": preds_all_seq,
        "targets_all": targets_all,
    }


# =========================
# Cross Validation
# =========================
def run_5fold_cv_meanpooled(
    df,
    X_count,
    bact_mean_all,
    mask_all,
    n_reads_all,
    make_model_fn,
    shared_ids,
    label_col="age",
    id_col="participant_id_index",
    row_idx_col="row_idx",
    n_splits=5,
    val_ratio=0.10,
    seed=42,
    batch_size=1,
    num_workers=0,
    lr=1e-4,
    weight_decay=0.01,
    epochs=10,
    device="cuda",
    scheduler_type="plateau",
    scheduler_patience=3,
    scheduler_factor=0.5,
    use_amp=True,
    amp_dtype=torch.float16,
    n_background=500,
    n_explain=500,
    feature_names=None,
    return_attn=False,
):
    # 只保留当前df中，属于shared_ids且label非空的样本
    shared_ids = set(shared_ids)
    df_use = df[df[id_col].isin(shared_ids)].copy()

    if label_col not in df_use.columns:
        raise ValueError(f"label_col='{label_col}' not found in df.columns")

    df_use = df_use[df_use[label_col].notna()].copy()

    if len(df_use) == 0:
        raise ValueError("No valid samples left after filtering by shared_ids and non-missing label.")

    # 再取一次当前df里实际可用的shared ids
    valid_shared_ids = np.array(sorted(df_use[id_col].astype(str).unique()), dtype=object)

    if len(valid_shared_ids) < n_splits:
        raise ValueError(
            f"Number of valid shared ids ({len(valid_shared_ids)}) is smaller than n_splits ({n_splits})."
        )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    all_fold_results = []
    all_pred_dfs = []
    all_pred_trains = []
    # all_shap_dfs = []
    all_shap_values = []
    all_feature_values = []
    all_sample_ids = []
    all_fold_ids = []
    background_pred_list = []
    w_list = []
    attention_list = []
    for fold, (trainval_idx, test_idx) in enumerate(kf.split(valid_shared_ids), start=1):
        trainval_ids = valid_shared_ids[trainval_idx]
        test_ids = valid_shared_ids[test_idx]

        train_ids, val_ids = train_test_split(
            trainval_ids,
            test_size=val_ratio,
            random_state=seed + fold,
            shuffle=True,
        )

        train_ids = set(train_ids.tolist())
        val_ids = set(val_ids.tolist())
        test_ids = set(test_ids.tolist())

        df_train = df_use[df_use[id_col].astype(str).isin(train_ids)].copy()
        df_val   = df_use[df_use[id_col].astype(str).isin(val_ids)].copy()
        df_test  = df_use[df_use[id_col].astype(str).isin(test_ids)].copy()

        train_ds = MeanPooledPersonDataset(
            df=df_train,
            X_count=X_count,
            bact_mean_all=bact_mean_all,
            mask_all=mask_all,
            n_reads_all=n_reads_all,
            label_col=label_col,
            id_col=id_col,
            filter_missing_label=True,
            row_idx_col=row_idx_col,
        )
        val_ds = MeanPooledPersonDataset(
            df=df_val,
            X_count=X_count,
            bact_mean_all=bact_mean_all,
            mask_all=mask_all,
            n_reads_all=n_reads_all,
            label_col=label_col,
            id_col=id_col,
            filter_missing_label=True,
            row_idx_col=row_idx_col,
        )
        test_ds = MeanPooledPersonDataset(
            df=df_test,
            X_count=X_count,
            bact_mean_all=bact_mean_all,
            mask_all=mask_all,
            n_reads_all=n_reads_all,
            label_col=label_col,
            id_col=id_col,
            filter_missing_label=True,
            row_idx_col=row_idx_col,
        )

        train_loader = make_loader_meanpooled(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            seed=seed + fold,
        )
        val_loader = make_loader_meanpooled(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=seed + 1000 + fold,
        )
        test_loader = make_loader_meanpooled(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=seed + 2000 + fold,
        )

        model = make_model_fn().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        if scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_factor,
                patience=scheduler_patience,
                min_lr=1e-7,
            )
        else:
            scheduler = None

        scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=(use_amp and device.startswith("cuda"))
        )

        best_val_mse = float("inf")
        best_val_pcc = -float("inf")
        best_state = None

        print(f"\n===== Fold {fold}/{n_splits} | label={label_col} =====")
        print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

        for epoch in range(1, epochs + 1):
            tr = train_one_epoch_meanpooled(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                epoch=epoch,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

            va = eval_one_epoch_meanpooled(
                model=model,
                loader=val_loader,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

            print(
                f"[Fold {fold}][Epoch {epoch}] "
                f"train_mse={tr['mse']:.4f} train_pcc={tr['pcc']:.4f} "
                f"val_mse={va['mse']:.4f} val_pcc={va['pcc']:.4f}"
            )

            if scheduler is not None:
                scheduler.step(va["mse"])

            if va["mse"] < best_val_mse:
                best_val_mse = va["mse"]
                best_val_pcc = va["pcc"]
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                print(
                    f"  ↳ best checkpoint updated by val_mse "
                    f"(val_mse={best_val_mse:.4f}, val_pcc={best_val_pcc:.4f})"
                )

        if best_state is not None:
            model.load_state_dict(best_state)

        te = eval_one_epoch_meanpooled(
            model=model,
            loader=test_loader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            return_attn=return_attn,
        )
        print(f"[Fold {fold}] TEST mse={te['mse']:.4f}  pcc={te['pcc']:.4f}")

        # shap
        model.eval()
        wrapped_model = CountOnlyWrapper(model, mode="weighted_count").to(device)
        wrapped_model.eval()

        # 不要用全 train_ds 当 background
        print(f"n_background: {n_background}, n_explain: {n_explain}")
        background_idx = sample_indices(
            n=len(train_ds),
            k=n_background,
            seed=seed + 3000 + fold,
        )

        explain_idx = sample_indices(
            n=len(test_ds),
            k=n_explain,
            seed=seed + 4000 + fold,
        )
        background = collect_count_tensor(train_ds, background_idx, device)
        x_explain = collect_count_tensor(test_ds, explain_idx, device)
        explainer = shap.GradientExplainer(wrapped_model, background)
        # explainer = shap.DeepExplainer(wrapped_model, background)
        background_pred_count = wrapped_model(background).mean().item()
        background_pred_list.append(background_pred_count)
        w_list.append(torch.sigmoid(model.alpha))
        attention_list.extend(te['atte_map_all'])
        t1 = time.time()
        shap_values = explainer.shap_values(x_explain)
        t2 = time.time()
        print(f"explaining time: {t2 - t1} seconds")

        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
            shap_values = shap_values[..., 0]
        x_explain_np = x_explain.detach().cpu().numpy()
        all_shap_values.append(shap_values.astype(np.float32))
        all_feature_values.append(x_explain_np.astype(np.float32))
        sample_ids = test_ds.df.iloc[explain_idx][id_col].astype(str).to_numpy()
        all_sample_ids.append(sample_ids)
        all_fold_ids.append(np.full(len(explain_idx), fold))

        print(f"[Fold {fold}] SHAP done: {shap_values.shape}")


        assert len(test_ds.base_indices) == len(te["preds_all"]) == len(te["targets_all"])

        fold_pred_df = pd.DataFrame({
            "idx": test_ds.base_indices,
            "participant_id_index": test_ds.df[id_col].astype(str).to_numpy(),
            "fold": fold,
            "pred": te["preds_all"],
            "pred_count": te["pred_count"],
            "pred_seq": te["pred_seq"],
            "target": te["targets_all"],
        })
        all_pred_dfs.append(fold_pred_df)

        all_fold_results.append({
            "label": label_col,
            "fold": fold,
            "best_val_pcc": best_val_pcc,
            "best_val_mse": best_val_mse,
            "test_mse": te["mse"],
            "test_rmse": te["rmse"],
            "test_mae": te["mae"],
            "test_pcc": te["pcc"],
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
        })

        fold_pred_train = pd.DataFrame({
            "idx": train_ds.base_indices,
            "participant_id_index": train_ds.df[id_col].astype(str).to_numpy(),
            "fold": fold,
            "pred": tr["preds_all"],
            "pred_count": tr["pred_count"],
            "pred_seq": tr["pred_seq"],
            "target": tr["targets_all"],
        })
        all_pred_trains.append(fold_pred_train)

    results_df = pd.DataFrame(all_fold_results)
    pred_df = pd.concat(all_pred_dfs, axis=0, ignore_index=True)
    pred_train = pd.concat(all_pred_trains, axis=0, ignore_index=True)

    shap_values_all = np.concatenate(all_shap_values, axis=0)
    feature_values_all = np.concatenate(all_feature_values, axis=0)
    sample_ids_all = np.concatenate(all_sample_ids, axis=0)
    fold_ids_all = np.concatenate(all_fold_ids, axis=0)
    sample_ids_fold = pd.DataFrame(data=[sample_ids_all, fold_ids_all]).T
    sample_ids_fold.columns = ['sample_ids_all', 'fold_ids_all']
    global_shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values_all).mean(axis=0),
        "mean_shap": shap_values_all.mean(axis=0),
    })
    global_shap_df = global_shap_df.sort_values(
        "mean_abs_shap",
        ascending=False,
    )
    shap.summary_plot(
        shap_values_all,
        features=feature_values_all,
        feature_names=feature_names,
        max_display=20,
    )


    print(global_shap_df.head(30))
    print("\n===== CV Summary =====")
    print(results_df)
    print("\nMean TEST PCC:", results_df["test_pcc"].mean())
    print("Std  TEST PCC:", results_df["test_pcc"].std())
    print("\nMean TEST MSE:", results_df["test_mse"].mean())
    print("Std  TEST MSE:", results_df["test_mse"].std())

    return results_df, pred_df, pred_train, shap_values_all, feature_values_all, global_shap_df, sample_ids_fold, background_pred_list, w_list, attention_list


def make_model_fn_meanpooled(count_dim):
    model = BacteriaMeanRegressor(
        d_model=512,
        n_bacteria=1000,
        count_dim=count_dim,
        count_hidden1=128,
        count_hidden2=16,
        bact_tf_layers=1,
        bact_tf_heads=8,
        mlp_ratio=4,
        dropout=0.0,
        sub_dim=4,
    )
    return model


# =========================
# Main
# =========================
if __name__ == "__main__":
    seed = 42
    seed_everything(seed, deterministic=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    adata = sc.read_h5ad("./AIDO_embedding/adata_bacteria_SGB_sort.h5ad")
    df = adata.obs.copy().reset_index(drop=True)
    df["person_index"] = 'person_' + df.index.astype(str)
    df["row_idx"] = np.arange(len(df), dtype=np.int64)
    df["npy_path"] = "./AIDO_embedding/save_by_individuals/" + df["person_index"] + ".npy"
    shared_ids = ['person_' + str(i) for i in range(0, 100)]
    # ===== 例子：构造多个表型 =====
    from sklearn.preprocessing import MinMaxScaler
    age_scaler = MinMaxScaler(feature_range=(0, 1))
    # df["age"] = age_scaler.fit_transform(df["Host_Age"].to_numpy().reshape(-1, 1)).astype(np.float32)
    df["age"] = df["age"] / 100
    # test !!!
    random_indices = df.sample(n=10).index
    df.loc[random_indices, 'age'] = np.nan

    # ===== count branch =====
    X_count = adata.X.toarray().astype(np.float32)

    # ===== mean pooled embedding =====
    MEAN_DIR = "./AIDO_embedding/meanpooled_by_person"
    meta_path = os.path.join(MEAN_DIR, "meta.parquet")
    if not os.path.exists(meta_path):
        raise NotImplementedError
    # load one time
    bact_mean_all = np.load(os.path.join(MEAN_DIR, "bact_mean.npy"))
    mask_all = np.load(os.path.join(MEAN_DIR, "mask.npy"))
    n_reads_all = np.load(os.path.join(MEAN_DIR, "n_reads.npy"))

    print("Loaded once:")
    print("  bact_mean_all:", bact_mean_all.shape, bact_mean_all.dtype)
    print("  mask_all     :", mask_all.shape, mask_all.dtype)
    print("  n_reads_all  :", n_reads_all.shape, n_reads_all.dtype)
    print("  X_count      :", X_count.shape, X_count.dtype)

    # ===== 你要循环预测的表型 =====
    phenotype_list = [
        "age",
        # "weight",
        # "bmi",
        # "sex",
    ]

    all_results = {}

    for phenotype in phenotype_list:
        print("\n" + "=" * 100)
        print(f"Start phenotype: {phenotype}")
        print("=" * 100)

        results_df, pred_df, pred_train, shap_values_all, feature_values_all, global_shap_df, sample_ids_fold, background_pred_list, w_list, attention_list = run_5fold_cv_meanpooled(df=df,
                                                    X_count=X_count,
                                                    bact_mean_all=bact_mean_all,
                                                    mask_all=mask_all,
                                                    n_reads_all=n_reads_all,
                                                    make_model_fn=lambda: make_model_fn_meanpooled(count_dim=X_count.shape[1]),
                                                    shared_ids=shared_ids,
                                                    label_col=phenotype,
                                                    id_col='person_index',  # participant_id_index
                                                    row_idx_col='row_idx',
                                                    n_splits=5,
                                                    val_ratio=0.10,
                                                    seed=seed,
                                                    batch_size=4,
                                                    num_workers=0,
                                                    lr=1e-4,
                                                    weight_decay=0.01,
                                                    epochs=5,
                                                    device=device,
                                                    scheduler_type="plateau",
                                                    scheduler_patience=3,
                                                    scheduler_factor=0.5,
                                                    use_amp=True,
                                                    amp_dtype=torch.float16,
                                                    n_background=500,
                                                    n_explain=500,
                                                    feature_names=adata.var.index.tolist(),
                                                    return_attn=True,
                                                )

        all_results[phenotype] = results_df
        results_df.to_csv(f"./results_{phenotype}.csv", index=False)
        pred_df.to_csv(f"./predictions_{phenotype}.csv", index=False)

    print("\nAll phenotype runs finished.")

#
pd.DataFrame(mask_all, index=df['person_index'], columns=adata.var.index.tolist()[0: 1000])