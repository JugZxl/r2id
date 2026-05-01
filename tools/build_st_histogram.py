"""Build uncertainty-aware spatio-temporal histograms for CLIP-ReID/R2ID."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import numpy as np


class UncertaintyAwareSTBuilder:
    def __init__(self):
        self.patterns = {
            'market': re.compile(r'(?P<pid>-?\d+)_c(?P<cam>\d+)s\d+_(?P<frame>-?\d+)_'),
            'duke': re.compile(r'(?P<pid>-?\d+)_c(?P<cam>\d+)_f(?P<frame>\d+)'),
            'msmt': re.compile(
                r'^(?P<pid>\d+)_(?P<idx>\d+)_(?P<cam>\d+)_(?P<date>\d{4})(?P<tod>morning|noon|afternoon|night)_(?P<frame>\d+)_(?P<tail>\d+)(?:_ex)?$',
                re.IGNORECASE,
            ),
        }

    @staticmethod
    def normalize_dataset_name(dataset: str) -> str:
        if dataset is None:
            return 'custom'
        ds = str(dataset).strip().lower().replace('-', '_')
        if ds in {'market1501', 'market_1501', 'market'}:
            return 'market'
        if ds in {'msmt17', 'msmt_17', 'msmt'}:
            return 'msmt'
        if ds in {'duke', 'dukemtmc', 'dukemtmc_reid', 'occ', 'occluded', 'occluded_duke', 'occluded_dukemtmc', 'occluded_duke_mtmc'}:
            return 'duke'
        if ds in {'veri', 'veri776'}:
            return 'veri'
        return ds

    def parse_filename(self, filename: str, dataset: str) -> Tuple[int, int, int]:
        basename = os.path.splitext(os.path.basename(filename))[0]
        dataset = self.normalize_dataset_name(dataset)
        if dataset == 'market':
            m = self.patterns['market'].search(basename)
            if m:
                return int(m.group('pid')), int(m.group('cam')) - 1, int(m.group('frame'))
        if dataset == 'duke':
            m = self.patterns['duke'].search(basename)
            if m:
                return int(m.group('pid')), int(m.group('cam')) - 1, int(m.group('frame'))
        if dataset == 'msmt':
            m = self.patterns['msmt'].search(basename)
            if m:
                tod_code = {'morning': 0, 'noon': 1, 'afternoon': 2, 'night': 3}
                pid = int(m.group('pid'))
                cam = int(m.group('cam')) - 1
                date = int(m.group('date'))
                tod = m.group('tod').lower()
                frame = int(m.group('frame'))
                time_val = date * 100000 + tod_code.get(tod, 0) * 10000 + frame
                return pid, cam, time_val
        # Generic fallback: pid, cam, last number as frame. Cam is assumed 1-based if >0.
        digits = re.findall(r'\d+', basename)
        if len(digits) >= 2:
            pid = int(digits[0])
            cam = max(int(digits[1]) - 1, 0)
            frame = int(digits[-1]) if len(digits) > 2 else 0
            return pid, cam, frame
        return -1, -1, 0

    def scan_train_dir(self, train_dir: str, dataset: str) -> Tuple[List[Tuple[int, int, int]], int]:
        samples = []
        cam_set = set()
        for root, _, files in os.walk(train_dir):
            for fname in files:
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                pid, cam, frame = self.parse_filename(fname, dataset)
                if pid >= 0 and cam >= 0:
                    samples.append((pid, cam, frame))
                    cam_set.add(cam)
        if not samples:
            raise ValueError(f'No valid images found in {train_dir}')
        return samples, max(cam_set) + 1 if cam_set else 1

    @staticmethod
    def compute_distribution_entropy(hist: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        C, C2, B = hist.shape
        assert C == C2, 'Histogram must be square in camera dimension'
        entropy = np.zeros((C, C), dtype=np.float32)
        for i in range(C):
            for j in range(C):
                if i == j:
                    continue
                if float(hist[i, j].sum()) <= eps:
                    entropy[i, j] = np.log(B + eps)
                    continue
                p = hist[i, j].astype(np.float32).copy()
                p = p / (p.sum() + eps)
                p = np.clip(p, eps, 1.0)
                entropy[i, j] = -np.sum(p * np.log(p))
        reliability = 1.0 - entropy / np.log(B + eps)
        return reliability.clip(0.0, 1.0)

    def build_uncertainty_aware_histogram(self, samples: List[Tuple[int, int, int]], num_cameras: int, dataset: str = 'custom', num_bins: int = 80, smooth_sigma_ratio: float = 0.5, percentile: float = 99.9) -> Dict[str, Any]:
        dataset = self.normalize_dataset_name(dataset)
        pid_cam_data: Dict[int, Dict[int, List[int]]] = {}
        for pid, cam, frame in samples:
            pid_cam_data.setdefault(pid, {}).setdefault(cam, []).append(frame)

        all_dt = []
        for cams in pid_cam_data.values():
            cam_list = list(cams.keys())
            if len(cam_list) < 2:
                continue
            for i in range(len(cam_list)):
                for j in range(i + 1, len(cam_list)):
                    for fi in cams[cam_list[i]]:
                        for fj in cams[cam_list[j]]:
                            dt = abs(fj - fi)
                            if dt > 0:
                                all_dt.append(dt)
        if not all_dt:
            raise ValueError('No cross-camera time differences found')

        dt_array = np.asarray(all_dt, dtype=np.float32)
        dt_max = max(float(np.percentile(dt_array, percentile)), 1.0)
        num_bins = max(10, int(num_bins))
        interval = int(math.ceil(dt_max / (num_bins - 1)))
        max_hist = interval * (num_bins - 1)
        smooth_sigma = smooth_sigma_ratio * interval
        print('[INFO] Time diff stats:')
        print(f'  - Min: {dt_array.min():.0f}, Max: {dt_array.max():.0f}')
        print(f'  - P{percentile}: {dt_max:.0f}, #pairs: {len(all_dt)}')
        print('[INFO] Bin params:')
        print(f'  - Num bins: {num_bins}, Interval: {interval}')
        print(f'  - Max hist: {max_hist}, Sigma: {smooth_sigma:.2f}')

        hist = np.zeros((num_cameras, num_cameras, num_bins), dtype=np.float64)
        for cams in pid_cam_data.values():
            cam_list = list(cams.keys())
            if len(cam_list) < 2:
                continue
            for i in range(len(cam_list)):
                for j in range(i + 1, len(cam_list)):
                    ci, cj = cam_list[i], cam_list[j]
                    for fi in cams[ci]:
                        for fj in cams[cj]:
                            dt = abs(fj - fi)
                            if dt <= 0:
                                continue
                            bin_idx = num_bins - 1 if dt >= max_hist else int(dt // interval)
                            hist[ci, cj, bin_idx] += 1.0
                            hist[cj, ci, bin_idx] += 1.0

        sigma_bins = smooth_sigma / float(interval) if interval > 0 else 0.0
        for i in range(num_cameras):
            for j in range(num_cameras):
                if i == j:
                    continue
                s = float(hist[i, j].sum())
                if s > 0:
                    if sigma_bins > 0:
                        hist[i, j] = self._gaussian_smooth(hist[i, j], sigma_bins)
                    hist[i, j] /= hist[i, j].sum() + 1e-12
                else:
                    hist[i, j] = 1.0 / float(num_bins)
        reliability = self.compute_distribution_entropy(hist.astype(np.float32))
        return {
            'distribution': hist.astype(np.float32),
            'reliability': reliability.astype(np.float32),
            'interval': interval,
            'max_hist': max_hist,
            'smooth_sigma': smooth_sigma,
            'num_bins': num_bins,
            'num_cameras': num_cameras,
            'dataset': dataset,
        }

    @staticmethod
    def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            return x
        k = int(max(3, math.ceil(6 * sigma)))
        if k % 2 == 0:
            k += 1
        c = k // 2
        xs = np.arange(-c, c + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (xs / sigma) ** 2)
        kernel /= kernel.sum() + 1e-12
        return np.convolve(x, kernel, mode='same')


def main():
    parser = argparse.ArgumentParser(description='Build uncertainty-aware spatio-temporal histogram')
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['market', 'market1501', 'duke', 'dukemtmc', 'occ', 'occluded', 'occluded_duke', 'occluded_dukemtmc', 'msmt', 'msmt17', 'veri', 'veri776', 'custom'])
    parser.add_argument('--output-npz', type=str, required=True)
    parser.add_argument('--num-bins', type=int, default=80)
    parser.add_argument('--smooth-ratio', type=float, default=0.5)
    parser.add_argument('--percentile', type=float, default=99.9)
    args = parser.parse_args()
    if not os.path.isdir(args.train_dir):
        print(f'Error: Training directory not found: {args.train_dir}')
        sys.exit(1)
    builder = UncertaintyAwareSTBuilder()
    print(f'Scanning {args.dataset} training directory: {args.train_dir}')
    samples, num_cameras = builder.scan_train_dir(args.train_dir, args.dataset)
    print(f'Found {len(samples)} samples, {num_cameras} cameras')
    print('Building uncertainty-aware spatio-temporal histogram...')
    result = builder.build_uncertainty_aware_histogram(samples=samples, num_cameras=num_cameras, dataset=args.dataset, num_bins=args.num_bins, smooth_sigma_ratio=args.smooth_ratio, percentile=args.percentile)
    os.makedirs(os.path.dirname(args.output_npz) or '.', exist_ok=True)
    np.savez_compressed(args.output_npz, **result)
    reliability = result['reliability']
    print(f'\n[SAVED] {args.output_npz}')
    print('[RELIABILITY STATS]')
    print(f'  Mean: {reliability.mean():.4f}')
    print(f'  Min: {reliability.min():.4f} pair={np.unravel_index(reliability.argmin(), reliability.shape)}')
    print(f'  Max: {reliability.max():.4f} pair={np.unravel_index(reliability.argmax(), reliability.shape)}')


if __name__ == '__main__':
    main()
