"""
R2ID inference plug-in for official CLIP-ReID.

Pipeline:
    visual feature extraction -> BIO -> UST-Fusion -> STG-AQE -> final UST-Fusion

This file is inference-only. It does not depend on model.stmeta or any training-time
feature fusion module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

try:
    from utils.st_histogram_fusion import load_distribution_npz
except Exception:
    load_distribution_npz = None

Tensor = torch.Tensor
ArrayLike = Union[np.ndarray, Sequence[int], Tensor]


def cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    cur = cfg
    for key in dotted.split('.'):
        if not hasattr(cur, key):
            return default
        cur = getattr(cur, key)
    return cur


def as_numpy_int(x: ArrayLike) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.int64)
    return np.asarray(x, dtype=np.int64)


def normalize_device_list(devices: Optional[Union[str, int, Iterable[Union[str, int]]]] = None) -> Sequence[str]:
    if devices is None:
        if torch.cuda.is_available():
            return [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        return ['cpu']

    if isinstance(devices, int):
        devices = [devices]
    elif isinstance(devices, str):
        s = devices.strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        devices = [p.strip().strip("'").strip('"') for p in s.split(',') if p.strip()]

    out = []
    for d in devices:  # type: ignore[union-attr]
        if isinstance(d, int):
            out.append(f'cuda:{d}')
        elif isinstance(d, str):
            d = d.strip()
            if d == 'cpu':
                out.append('cpu')
            elif d == 'cuda':
                out.append('cuda:0')
            elif d.startswith('cuda:'):
                out.append(d)
            elif d.isdigit():
                out.append(f'cuda:{d}')
    if not out:
        out = ['cuda:0'] if torch.cuda.is_available() else ['cpu']
    if not torch.cuda.is_available():
        return ['cpu']
    return out


@dataclass
class R2IDPluginConfig:
    # BIO
    use_bio: bool = True
    bio_topk: int = 6
    bio_base_strength: float = 0.05
    bio_power: float = 2.0
    bio_temperature: float = 0.05
    bio_min_difficulty: float = 0.1
    bio_max_difficulty: float = 0.9

    # UST-Fusion
    use_st: bool = True
    npz_path: str = ''
    lambda_st: float = 0.13
    uncertainty_method: str = 'sqrt'  # fixed / linear / sqrt / sigmoid / piecewise / boosted_linear
    reliability_min: float = 0.4
    reliability_floor: float = 0.1
    lambda_boost: float = 1.0
    sigmoid_k: float = 5.0
    piecewise_thresholds: Tuple[float, float] = (0.4, 0.7)
    piecewise_weights: Tuple[float, float, float] = (0.4, 0.7, 1.0)

    # STG-AQE
    use_stg_aqe: bool = True
    aqe_topk: int = 8
    aqe_alpha: float = 0.8
    aqe_tau: float = 0.1
    aqe_use_reliability_smoothing: bool = True
    aqe_eta0: float = 0.5

    # Runtime
    apply_final_ust: bool = True
    chunk_size: int = 128
    devices: Optional[Union[str, int, Iterable[Union[str, int]]]] = None
    out_dtype: torch.dtype = torch.float16

    @classmethod
    def from_cfg(cls, cfg: Any) -> 'R2IDPluginConfig':
        npz_path = str(cfg_get(cfg, 'META.ST_HIST.NPZ_PATH', '') or '')
        use_st = bool(cfg_get(cfg, 'META.ST_HIST.ENABLE', False)) or bool(npz_path)
        return cls(
            use_bio=bool(cfg_get(cfg, 'BIO.ENHANCE.ENABLE', True)),
            bio_topk=int(cfg_get(cfg, 'BIO.ENHANCE.TOPK', 6)),
            bio_base_strength=float(cfg_get(cfg, 'BIO.ENHANCE.BASE_STRENGTH', 0.05)),
            bio_power=float(cfg_get(cfg, 'BIO.ENHANCE.POWER', 2.0)),
            bio_temperature=float(cfg_get(cfg, 'BIO.ENHANCE.TEMPERATURE', 0.05)),
            use_st=use_st,
            npz_path=npz_path,
            lambda_st=float(cfg_get(cfg, 'META.ST_HIST.LAMBDA', cfg_get(cfg, 'ST.LAMBDA', 0.13))),
            uncertainty_method=str(cfg_get(cfg, 'ST.UNCERTAINTY_METHOD', 'sqrt')),
            reliability_min=float(cfg_get(cfg, 'ST.RELIABILITY_MIN', 0.4)),
            reliability_floor=float(cfg_get(cfg, 'ST.RELIABILITY_FLOOR', 0.1)),
            lambda_boost=float(cfg_get(cfg, 'ST.LAMBDA_BOOST', 1.0)),
            sigmoid_k=float(cfg_get(cfg, 'ST.SIGMOID_K', 5.0)),
            use_stg_aqe=bool(cfg_get(cfg, 'META.AQE.ST_GUIDED', True)),
            aqe_topk=int(cfg_get(cfg, 'META.AQE.TOPK', 8)),
            aqe_alpha=float(cfg_get(cfg, 'META.AQE.ALPHA', 0.8)),
            aqe_tau=float(cfg_get(cfg, 'META.AQE.TAU', 0.1)),
            aqe_use_reliability_smoothing=bool(cfg_get(cfg, 'META.AQE.RELIABILITY_SMOOTH', True)),
            aqe_eta0=float(cfg_get(cfg, 'META.AQE.ETA0', 0.5)),
            apply_final_ust=bool(cfg_get(cfg, 'META.ST_HIST.FINAL_FUSION', True)),
            chunk_size=int(cfg_get(cfg, 'ST.BLOCKWISE_CHUNK_SIZE', 128)),
            devices=cfg_get(cfg, 'ST.BLOCKWISE_DEVICES', None),
        )


def fixed_st_method(method: str) -> bool:
    return str(method).lower() in {'fixed', 'constant', 'none', 'static', 'fixed_st'}


def adaptive_lambda(
    R: Tensor,
    lambda_st: float,
    method: str = 'sqrt',
    reliability_min: float = 0.4,
    reliability_floor: float = 0.1,
    lambda_boost: float = 1.0,
    sigmoid_k: float = 5.0,
    piecewise_thresholds: Tuple[float, float] = (0.4, 0.7),
    piecewise_weights: Tuple[float, float, float] = (0.4, 0.7, 1.0),
) -> Tensor:
    method = str(method).lower()
    if fixed_st_method(method):
        lam = torch.full_like(R, float(lambda_st))
    else:
        R = torch.clamp(R.float(), reliability_min, 1.0)
        if method == 'linear':
            lam = lambda_st * R
        elif method == 'sqrt':
            lam = lambda_st * torch.sqrt(R + 1e-8)
        elif method == 'sigmoid':
            lam = lambda_st * torch.sigmoid(sigmoid_k * (R - 0.5))
        elif method == 'piecewise':
            lam = torch.zeros_like(R)
            t0, t1 = piecewise_thresholds
            w0, w1, w2 = piecewise_weights
            lam[R < t0] = lambda_st * w0
            lam[(R >= t0) & (R < t1)] = lambda_st * w1
            lam[R >= t1] = lambda_st * w2
        elif method == 'boosted_linear':
            lam = lambda_st * R * lambda_boost
        else:
            lam = lambda_st * R
    lam = torch.clamp(lam, 0.0, 1.0)
    return torch.maximum(lam, torch.full_like(lam, reliability_floor * lambda_st))


def bio_enhance(
    q_feats: Tensor,
    g_feats: Tensor,
    sim_mat: Tensor,
    topk: int = 6,
    base_strength: float = 0.05,
    power: float = 2.0,
    temperature: float = 0.05,
    min_difficulty: float = 0.1,
    max_difficulty: float = 0.9,
) -> Tuple[Tensor, Dict[str, float]]:
    Q, _ = q_feats.shape
    G = g_feats.shape[0]
    if Q == 0 or G == 0 or topk <= 0:
        return q_feats, {'bio_change': 0.0, 'difficulty_mean': 0.0, 'strength_mean': 0.0}

    k = min(int(topk), G)
    topk_sims, topk_idxs = torch.topk(sim_mat, k=k, dim=1)
    sim_mean = topk_sims.mean(dim=1)
    sim_var = topk_sims.var(dim=1, unbiased=False)
    eps = 1e-8
    mean_span = sim_mean.max() - sim_mean.min()
    var_span = sim_var.max() - sim_var.min()
    norm_mean = torch.zeros_like(sim_mean) if mean_span.abs() < eps else (sim_mean - sim_mean.min()) / (mean_span + eps)
    norm_var = torch.zeros_like(sim_var) if var_span.abs() < eps else (sim_var - sim_var.min()) / (var_span + eps)
    difficulty = ((1.0 - norm_mean) * (1.0 - norm_var)).clamp(min_difficulty, max_difficulty)
    strength = base_strength * (0.7 + 0.6 * difficulty) * torch.sigmoid(power * (difficulty - 0.5))

    weights = F.softmax(topk_sims / max(float(temperature), 1e-6), dim=1)
    neigh = g_feats[topk_idxs]
    direction = torch.sum((neigh - q_feats.unsqueeze(1)) * weights.unsqueeze(-1), dim=1)
    direction = F.normalize(direction, p=2, dim=1)
    out = F.normalize(q_feats + strength.unsqueeze(1) * direction, p=2, dim=1)
    return out, {
        'bio_change': float(torch.norm(out - q_feats, dim=1).mean().item()),
        'difficulty_mean': float(difficulty.mean().item()),
        'strength_mean': float(strength.mean().item()),
    }


class R2IDInferencePlugin:
    def __init__(self, config: R2IDPluginConfig, st_model: Optional[Any] = None, device: Optional[str] = None):
        self.cfg = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.devices = normalize_device_list(config.devices)
        self.logger = logging.getLogger('r2id.plugin')
        if st_model is not None:
            self.st_model = st_model
        elif config.use_st and config.npz_path:
            if load_distribution_npz is None:
                raise ImportError('utils.st_histogram_fusion.load_distribution_npz is unavailable.')
            self.st_model = load_distribution_npz(config.npz_path, device=self.device)
        else:
            self.st_model = None

    @classmethod
    def from_cfg(cls, cfg: Any, device: Optional[str] = None, st_model: Optional[Any] = None) -> 'R2IDInferencePlugin':
        return cls(R2IDPluginConfig.from_cfg(cfg), st_model=st_model, device=device)

    def available_st(self) -> bool:
        return bool(self.cfg.use_st and self.st_model is not None and self.cfg.lambda_st > 0.0)

    @staticmethod
    def visual_sim(q_feats: Tensor, g_feats: Tensor) -> Tensor:
        return F.normalize(q_feats.float(), p=2, dim=1) @ F.normalize(g_feats.float(), p=2, dim=1).t()

    def st_prob_and_lambda(self, q_camids: ArrayLike, g_camids: ArrayLike, q_frames: ArrayLike, g_frames: ArrayLike, dev: str, q_slice: Optional[slice] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if self.st_model is None:
            raise RuntimeError('ST model is not loaded.')
        q_cam_np = as_numpy_int(q_camids)
        q_frm_np = as_numpy_int(q_frames)
        g_cam_np = as_numpy_int(g_camids)
        g_frm_np = as_numpy_int(g_frames)
        if q_slice is not None:
            q_cam_np = q_cam_np[q_slice]
            q_frm_np = q_frm_np[q_slice]

        q_cam = torch.as_tensor(q_cam_np, dtype=torch.long, device=dev)
        q_frm = torch.as_tensor(q_frm_np, dtype=torch.long, device=dev)
        g_cam = torch.as_tensor(g_cam_np, dtype=torch.long, device=dev)
        g_frm = torch.as_tensor(g_frm_np, dtype=torch.long, device=dev)
        distribution = self.st_model.distribution.to(dev)
        interval = int(self.st_model.interval)
        num_bins = int(self.st_model.num_bins)
        time_diff = torch.abs(q_frm.unsqueeze(1) - g_frm.unsqueeze(0))
        bin_idx = torch.clamp(time_diff // max(interval, 1), 0, num_bins - 1).long()
        q_cam_exp = q_cam.unsqueeze(1).expand(-1, g_cam.numel())
        g_cam_exp = g_cam.unsqueeze(0).expand(q_cam.numel(), -1)
        st_probs = distribution[q_cam_exp, g_cam_exp, bin_idx].float()
        lam = None
        reliability = getattr(self.st_model, 'reliability', None)
        if reliability is not None:
            R = reliability.to(dev)[q_cam_exp, g_cam_exp].float()
            lam = adaptive_lambda(
                R,
                lambda_st=self.cfg.lambda_st,
                method=self.cfg.uncertainty_method,
                reliability_min=self.cfg.reliability_min,
                reliability_floor=self.cfg.reliability_floor,
                lambda_boost=self.cfg.lambda_boost,
                sigmoid_k=self.cfg.sigmoid_k,
                piecewise_thresholds=self.cfg.piecewise_thresholds,
                piecewise_weights=self.cfg.piecewise_weights,
            )
        return st_probs, lam

    def ust_fuse_blockwise(self, q_feats: Tensor, g_feats: Tensor, q_camids: ArrayLike, g_camids: ArrayLike, q_frames: ArrayLike, g_frames: ArrayLike, return_sim: bool = True) -> Tuple[Tensor, Dict[str, float]]:
        Q, G = q_feats.shape[0], g_feats.shape[0]
        out = torch.empty((Q, G), dtype=self.cfg.out_dtype, device='cpu')
        acc = {'sum': 0.0, 'sq_sum': 0.0, 'count': 0.0, 'min': float('inf'), 'max': float('-inf')}
        q_cpu = F.normalize(q_feats.detach().cpu().float(), p=2, dim=1).contiguous()
        g_cpu = F.normalize(g_feats.detach().cpu().float(), p=2, dim=1).contiguous()
        per_dev: Dict[str, Dict[str, Tensor]] = {}
        for dev in self.devices:
            g_dev = g_cpu.to(dev, non_blocking=True)
            per_dev[dev] = {'g': g_dev, 'gt': g_dev.t().contiguous()}
        with torch.no_grad():
            for block_id, start in enumerate(range(0, Q, self.cfg.chunk_size)):
                end = min(start + self.cfg.chunk_size, Q)
                dev = self.devices[block_id % len(self.devices)]
                q_dev = q_cpu[start:end].to(dev, non_blocking=True)
                sim = q_dev @ per_dev[dev]['gt']
                if self.available_st():
                    st_probs, lam = self.st_prob_and_lambda(q_camids, g_camids, q_frames, g_frames, dev=dev, q_slice=slice(start, end))
                    if lam is None:
                        fused = (1.0 - self.cfg.lambda_st) * sim + self.cfg.lambda_st * st_probs
                    else:
                        fused = (1.0 - lam) * sim + lam * st_probs
                        lam_f = lam.detach().float()
                        acc['sum'] += float(lam_f.sum().item())
                        acc['sq_sum'] += float((lam_f ** 2).sum().item())
                        acc['count'] += float(lam_f.numel())
                        acc['min'] = min(acc['min'], float(lam_f.min().item()))
                        acc['max'] = max(acc['max'], float(lam_f.max().item()))
                    block = fused if return_sim else (1.0 - fused)
                else:
                    block = sim if return_sim else (1.0 - sim)
                out[start:end] = block.detach().to('cpu', dtype=self.cfg.out_dtype)
                del q_dev, sim, block
                if dev != 'cpu':
                    torch.cuda.empty_cache()
        stats: Dict[str, float] = {}
        if acc['count'] > 0:
            mean = acc['sum'] / acc['count']
            var = max(acc['sq_sum'] / acc['count'] - mean * mean, 0.0)
            stats = {'lambda_mean': mean, 'lambda_std': var ** 0.5, 'lambda_min': acc['min'], 'lambda_max': acc['max']}
        return out, stats

    def st_guided_aqe(self, q_feats: Tensor, g_feats: Tensor, candidate_sim: Tensor, q_camids: ArrayLike, g_camids: ArrayLike, q_frames: ArrayLike, g_frames: ArrayLike) -> Tensor:
        if not self.available_st() or not self.cfg.use_stg_aqe or self.cfg.aqe_topk <= 0 or self.cfg.aqe_alpha <= 0:
            return q_feats
        device = q_feats.device
        Q, G = candidate_sim.shape
        k = min(int(self.cfg.aqe_topk), G)
        candidate_sim_dev = candidate_sim.to(device=device, dtype=torch.float32)
        topk_sims, topk_idx = torch.topk(candidate_sim_dev, k=k, dim=1)
        q_cam_np = as_numpy_int(q_camids)
        q_frm_np = as_numpy_int(q_frames)
        g_cam_np = as_numpy_int(g_camids)
        g_frm_np = as_numpy_int(g_frames)
        flat_idx = topk_idx.detach().cpu().numpy().reshape(-1)
        q_cam_rep = np.repeat(q_cam_np, k)
        q_frm_rep = np.repeat(q_frm_np, k)
        g_cam_sel = g_cam_np[flat_idx]
        g_frm_sel = g_frm_np[flat_idx]
        hist = self.st_model.distribution.detach().cpu().numpy()
        interval = int(self.st_model.interval)
        num_bins = int(self.st_model.num_bins)
        bin_idx = np.clip(np.abs(q_frm_rep - g_frm_sel) // max(interval, 1), 0, num_bins - 1).astype(np.int64)
        st_vals = hist[q_cam_rep, g_cam_sel, bin_idx].astype(np.float32)
        st_probs = torch.from_numpy(st_vals).to(device).view(Q, k)
        if self.cfg.aqe_use_reliability_smoothing and getattr(self.st_model, 'reliability', None) is not None:
            R = self.st_model.reliability.detach().cpu().numpy().astype(np.float32)
            eta_vals = self.cfg.aqe_eta0 * (1.0 - R[q_cam_rep, g_cam_sel])
            eta = torch.from_numpy(eta_vals.astype(np.float32)).to(device).view(Q, k)
            st_probs = (1.0 - eta) * st_probs + eta * (1.0 / float(num_bins))
        joint = topk_sims * st_probs
        weights = F.softmax(joint / max(float(self.cfg.aqe_tau), 1e-6), dim=1)
        neigh_feats = g_feats[topk_idx]
        expanded = torch.sum(neigh_feats * weights.unsqueeze(-1), dim=1)
        return F.normalize(q_feats + float(self.cfg.aqe_alpha) * expanded, p=2, dim=1)

    def __call__(self, q_feats: Tensor, g_feats: Tensor, q_camids: ArrayLike, g_camids: ArrayLike, q_frames: ArrayLike, g_frames: ArrayLike) -> Tuple[Tensor, Dict[str, float]]:
        info: Dict[str, float] = {}
        q = F.normalize(q_feats.to(self.device).float(), p=2, dim=1)
        g = F.normalize(g_feats.to(self.device).float(), p=2, dim=1)
        sim0 = self.visual_sim(q, g)
        if self.cfg.use_bio:
            q, bio_info = bio_enhance(q, g, sim0, topk=self.cfg.bio_topk, base_strength=self.cfg.bio_base_strength, power=self.cfg.bio_power, temperature=self.cfg.bio_temperature, min_difficulty=self.cfg.bio_min_difficulty, max_difficulty=self.cfg.bio_max_difficulty)
            info.update(bio_info)
        if self.available_st():
            sim_candidate, ust_info = self.ust_fuse_blockwise(q, g, q_camids, g_camids, q_frames, g_frames, return_sim=True)
            info.update({f'candidate_{k}': v for k, v in ust_info.items()})
        else:
            sim_candidate = self.visual_sim(q, g).detach().cpu().to(self.cfg.out_dtype)
        q_final = self.st_guided_aqe(q, g, sim_candidate, q_camids, g_camids, q_frames, g_frames)
        if self.cfg.apply_final_ust and self.available_st():
            dist, final_info = self.ust_fuse_blockwise(q_final, g, q_camids, g_camids, q_frames, g_frames, return_sim=False)
            info.update({f'final_{k}': v for k, v in final_info.items()})
        else:
            sim_final = self.visual_sim(q_final, g)
            dist = (1.0 - sim_final).detach().cpu().to(self.cfg.out_dtype)
        return dist, info
