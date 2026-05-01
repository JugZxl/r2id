"""ST histogram loading and frame parsing utilities for R2ID."""

from __future__ import annotations

import math
import os
import re
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    'STHistogramFusion',
    'load_distribution_npz',
    'parse_frame_ids',
    'compute_camera_pair_reliability',
    'gaussian_smooth_1d',
]


def parse_frame_ids(paths):
    """Parse comparable temporal indices from image paths.

    Supports:
      - Market-1501: 0002_c1s1_000451_03.jpg -> 451
      - Duke/Occluded-Duke: xxxx_c2_f0046985.jpg -> 46985
      - MSMT17 variants: 0080_035_05_0303afternoon_1084_1.jpg
        -> date*100000 + tod_code*10000 + frame
    """
    frames = []
    msmt_pat = re.compile(
        r'^(?P<pid>\d+)_(?P<idx>\d+)_(?P<cam>\d+)_(?P<date>\d{4})(?P<tod>morning|noon|afternoon|night)_(?P<frame>\d+)_(?P<tail>\d+)(?:_ex)?$',
        re.IGNORECASE,
    )
    tod_code = {'morning': 0, 'noon': 1, 'afternoon': 2, 'night': 3}
    patterns = [
        r'_c\d+s\d+_(\d+)_',  # Market
        r'_c\d+_f(\d+)',       # Duke / Occluded-Duke
        r'f(\d+)',             # fallback Duke-like
        r'frame_?(\d+)',
        r'_(\d{6,})\.',
        r'(\d{6,})\.jpg',
        r'_(\d{4,})$',
    ]
    for path in paths:
        name = os.path.splitext(os.path.basename(str(path)))[0]
        m = msmt_pat.search(name)
        if m:
            date = int(m.group('date'))
            tod = m.group('tod').lower()
            frame = int(m.group('frame'))
            frames.append(date * 100000 + tod_code.get(tod, 0) * 10000 + frame)
            continue
        frame = 0
        filename = os.path.basename(str(path))
        for pat in patterns:
            mm = re.search(pat, filename, flags=re.IGNORECASE)
            if mm:
                frame = int(mm.group(1))
                break
        frames.append(frame)
    return np.asarray(frames, dtype=np.int64)


def gaussian_smooth_1d(x: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return x
    k = int(max(3, math.ceil(6 * sigma_bins)))
    if k % 2 == 0:
        k += 1
    center = k // 2
    xs = np.arange(-center, center + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (xs / sigma_bins) ** 2)
    kernel /= kernel.sum() + 1e-12
    return np.convolve(x, kernel, mode='same')


def compute_camera_pair_reliability(distribution: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    C, _, B = distribution.shape
    reliability = np.ones((C, C), dtype=np.float32)
    logB = np.log(B + eps)
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            p = distribution[i, j].astype(np.float32).copy()
            s = float(p.sum())
            if s <= eps:
                reliability[i, j] = 0.0
                continue
            p = p / (s + eps)
            p = np.clip(p, eps, 1.0)
            entropy = -np.sum(p * np.log(p))
            reliability[i, j] = 1.0 - entropy / logB
    return np.clip(reliability, 0.0, 1.0)


class STHistogramFusion(nn.Module):
    def __init__(self, interval: int = 100, max_hist: int = 5000, smooth_sigma: float = 50.0, num_cameras: Optional[int] = None, device: str = 'cpu', num_bins: Optional[int] = None, **kwargs):
        super().__init__()
        self.interval = int(interval)
        self.max_hist = int(max_hist)
        self.smooth_sigma = float(smooth_sigma)
        self.num_cameras = int(num_cameras) if num_cameras is not None else None
        self.num_bins = int(num_bins) if num_bins is not None else None
        self.device = torch.device(device)
        self._distribution = None
        self._reliability = None

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, value):
        if isinstance(value, np.ndarray):
            self._distribution = torch.from_numpy(value).float().to(self.device)
        elif isinstance(value, torch.Tensor):
            self._distribution = value.float().to(self.device)
        else:
            raise ValueError(f'Unsupported distribution type: {type(value)}')
        if self.num_cameras is None and self._distribution is not None:
            self.num_cameras = self._distribution.shape[0]
        if self.num_bins is None and self._distribution is not None:
            self.num_bins = self._distribution.shape[2]

    @property
    def reliability(self):
        return self._reliability

    @reliability.setter
    def reliability(self, value):
        if value is None:
            self._reliability = None
        elif isinstance(value, np.ndarray):
            self._reliability = torch.from_numpy(value).float().to(self.device)
        elif isinstance(value, torch.Tensor):
            self._reliability = value.float().to(self.device)
        else:
            raise ValueError(f'Unsupported reliability type: {type(value)}')

    def to(self, device=None, dtype=None, non_blocking=False):
        if device is not None:
            self.device = torch.device(device)
        super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        if self._distribution is not None:
            self._distribution = self._distribution.to(self.device)
        if self._reliability is not None:
            self._reliability = self._reliability.to(self.device)
        return self

    def get_st_probability(self, q_cam: torch.Tensor, q_frame: torch.Tensor, g_cam: torch.Tensor, g_frame: torch.Tensor, use_reliability: bool = False, reliability_floor: float = 0.1) -> torch.Tensor:
        if self._distribution is None:
            raise RuntimeError('Distribution not loaded')
        device = self._distribution.device
        q_cam = q_cam.to(device).long()
        q_frame = q_frame.to(device).long()
        g_cam = g_cam.to(device).long()
        g_frame = g_frame.to(device).long()
        time_diff = torch.abs(q_frame.unsqueeze(1) - g_frame.unsqueeze(0))
        bin_idx = torch.clamp((time_diff // max(int(self.interval), 1)).long(), 0, int(self.num_bins) - 1)
        q_cam_exp = q_cam.unsqueeze(1).expand(-1, g_cam.size(0))
        g_cam_exp = g_cam.unsqueeze(0).expand(q_cam.size(0), -1)
        st_probs = self._distribution[q_cam_exp, g_cam_exp, bin_idx]
        if use_reliability and self._reliability is not None:
            pair_reliability = self._reliability[q_cam_exp, g_cam_exp]
            uniform = 1.0 / float(self.num_bins)
            w = torch.clamp(pair_reliability, reliability_floor, 1.0)
            st_probs = w * st_probs + (1.0 - w) * uniform
        return st_probs


def load_distribution_npz(npz_path: str, device: str = 'cpu') -> STHistogramFusion:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f'ST histogram file not found: {npz_path}')
    data = np.load(npz_path, allow_pickle=True)
    if 'distribution' not in data:
        raise ValueError(f"'distribution' not found in {npz_path}")
    distribution = data['distribution'].astype(np.float32)
    interval = int(data['interval']) if 'interval' in data else 100
    max_hist = int(data['max_hist']) if 'max_hist' in data else (distribution.shape[2] - 1) * interval
    smooth_sigma = float(data['smooth_sigma']) if 'smooth_sigma' in data else 50.0
    num_cameras = int(data['num_cameras']) if 'num_cameras' in data else distribution.shape[0]
    num_bins = int(data['num_bins']) if 'num_bins' in data else distribution.shape[2]
    model = STHistogramFusion(interval=interval, max_hist=max_hist, smooth_sigma=smooth_sigma, num_cameras=num_cameras, num_bins=num_bins, device=device)
    model.distribution = distribution
    if 'reliability' in data:
        model.reliability = data['reliability'].astype(np.float32)
        print(f'[ST Model] Loaded reliability matrix: mean={data["reliability"].mean():.3f}')
    print(f'[ST Model] Loaded from {npz_path}')
    print(f'  - Distribution shape: {distribution.shape}')
    print(f'  - Interval: {interval}, Max hist: {max_hist}')
    print(f'  - Num cameras: {num_cameras}, Num bins: {num_bins}')
    return model
