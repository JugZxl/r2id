import os
import re
import sys
import math
import argparse

from typing import List, Tuple, Dict, Any

import numpy as np


class UncertaintyAwareSTBuilder:
    
    def __init__(self):
        self.patterns = {
            'market': re.compile(r"([-\d]+)_c(\d+)s(\d+)_([-\d]+)_([-\d]+)"),
            'duke': re.compile(r"([-\d]+)_c(\d+)_f(\d+)"),
            # 'msmt': re.compile(r"([-\d]+)_c(\d+)_f(\d+)"),
            'msmt' : re.compile(r"(?P<pid>\d+)_(?P<cam>\d{3})_(?P<seq>\d{2})_(?P<date>\d{4})(?P<tod>morning|afternoon|night)_(?P<frame>\d{4})_(?P<idx>\d+)(?:_ex)?$"
),

        }

    @staticmethod
    def normalize_dataset_name(dataset: str) -> str:
        """统一数据集命名（兼容常见别名）"""
        if dataset is None:
            return 'custom'
        ds = str(dataset).strip().lower()
        ds = ds.replace('-', '_')

        # 常见别名
        if ds in {'market1501', 'market_1501', 'market'}:
            return 'market'
        if ds in {'msmt17', 'msmt_17', 'msmt'}:
            return 'msmt'
        if ds in {
            'duke', 'dukemtmc', 'dukemtmc_reid',
            'occ', 'occluded', 'occluded_duke', 'occluded_dukemtmc',
            'occluded_duke_mtmc'
        }:
            # Occluded-DukeMTMC 的文件命名/摄像头编号通常沿用 Duke
            return 'duke'
        if ds in {'veri', 'veri776'}:
            return 'veri'
        return ds
    
    def parse_filename(self, filename: str, dataset: str) -> Tuple[int, int, int]:
        """解析文件名，返回(pid, cam, frame)"""
        basename = os.path.splitext(os.path.basename(filename))[0]
        dataset = self.normalize_dataset_name(dataset)
        
        if dataset == 'market':
            m = self.patterns['market'].search(basename)
            if m:
                try:
                    pid = int(m.group(1))
                    # Market-1501 cam 是 1-based，统一转为 0-based
                    cam = int(m.group(2)) - 1
                    frame = int(m.group(4))  # market的帧号在第4组
                    return pid, cam, frame
                except:
                    pass
        
        elif dataset == 'duke':
            m = self.patterns['duke'].search(basename)
            if m:
                try:
                    pid = int(m.group(1))
                    cam = int(m.group(2)) - 1  # Duke是1-based，转为0-based
                    frame = int(m.group(3))
                    return pid, cam, frame
                except:
                    pass
        
        elif dataset == 'msmt':
            m = self.patterns['msmt'].search(basename)
            if m:
                try:
                    pid = int(m.group(1))
                    # MSMT17 cam 通常也是 1-based（c1..c15），统一转为 0-based
                    cam = int(m.group(2)) - 1
                    frame = int(m.group(3))
                    return pid, cam, frame
                except:
                    pass
        elif dataset == 'msmt':
            m = self.patterns['msmt'].search(basename)
            if m:
                pid = int(m.group('pid'))
                cam = int(m.group('cam')) - 1     # 001..015 -> 0..14
                date = int(m.group('date'))       # 0303
                tod = m.group('tod')
                frame = int(m.group('frame'))     # 0035

                tod_code = {'morning':0, 'afternoon':1, 'night':2}[tod]
                # 构造一个“跨cam可比较”的时间轴（同一天同时间段同frame可比）
                t = date * 100000 + tod_code * 10000 + frame
                return pid, cam, t
            
        # 通用模式：尝试匹配末尾的数字作为帧号
        digits = re.findall(r'\d+', basename)
        if len(digits) >= 2:
            try:
                pid = int(digits[0])
                cam = int(digits[1])
                frame = int(digits[-1]) if len(digits) > 2 else 0
                return pid, cam, frame
            except:
                pass
        
        return -1, -1, 0
    
    def scan_train_dir(self, train_dir: str, dataset: str) -> Tuple[List[Tuple[int, int, int]], int]:
        """扫描训练目录，收集所有样本"""
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
            raise ValueError(f"No valid images found in {train_dir}")
        
        num_cameras = max(cam_set) + 1 if cam_set else 1
        return samples, num_cameras
    
    def compute_distribution_entropy(self, hist: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """计算分布的熵（不确定性）"""
        C, C2, B = hist.shape
        assert C == C2, "Histogram must be square in camera dimension"
        
        # 计算熵 H = -∑ p * log(p)
        entropy = np.zeros((C, C), dtype=np.float32)
        
        for i in range(C):
            for j in range(C):
                if i == j:
                    continue
                # 没有统计到该摄像头对：直接判为完全不可靠
                if float(hist[i, j].sum()) <= eps:
                    entropy[i, j] = np.log(B + eps)  # 对应可靠性=0
                    continue
                p = hist[i, j].copy()
                p = p.clip(eps, 1.0)
                h = -np.sum(p * np.log(p))
                entropy[i, j] = h
        
        # 归一化到[0,1]
        logB = np.log(B + eps)
        normalized_entropy = entropy / logB
        reliability = 1.0 - normalized_entropy  # 可靠性 = 1 - 归一化熵
        
        return reliability.clip(0.0, 1.0)
    
    def build_uncertainty_aware_histogram(
        self,
        samples: List[Tuple[int, int, int]],
        num_cameras: int,
        dataset: str = 'custom',
        num_bins: int = 80,
        smooth_sigma_ratio: float = 0.5,
        percentile: float = 99.9
    ) -> Dict[str, Any]:
        """构建不确定性感知的时空直方图"""

        dataset = self.normalize_dataset_name(dataset)
        
        # 1. 按pid和cam组织数据
        pid_cam_data: Dict[int, Dict[int, List[int]]] = {}
        for pid, cam, frame in samples:
            if pid not in pid_cam_data:
                pid_cam_data[pid] = {}
            if cam not in pid_cam_data[pid]:
                pid_cam_data[pid][cam] = []
            pid_cam_data[pid][cam].append(frame)
        
        # 2. 收集所有时间差用于自适应bin选择
        all_dt = []
        for pid, cams in pid_cam_data.items():
            cam_list = list(cams.keys())
            if len(cam_list) < 2:
                continue
            
            for i in range(len(cam_list)):
                for j in range(i + 1, len(cam_list)):
                    cam_i, cam_j = cam_list[i], cam_list[j]
                    for frame_i in cams[cam_i]:
                        for frame_j in cams[cam_j]:
                            dt = abs(frame_j - frame_i)
                            if dt > 0:
                                all_dt.append(dt)
        
        if not all_dt:
            raise ValueError("No cross-camera time differences found")
        
        # 3. 自适应确定bin参数
        dt_array = np.array(all_dt, dtype=np.float32)
        dt_max = np.percentile(dt_array, percentile)
        dt_max = max(dt_max, 1.0)
        
        # 确定bin数量和宽度
        num_bins = max(10, int(num_bins))
        interval = int(math.ceil(dt_max / (num_bins - 1)))
        max_hist = interval * (num_bins - 1)
        smooth_sigma = smooth_sigma_ratio * interval
        
        print(f"[INFO] Time diff stats:")
        print(f"  - Min: {dt_array.min():.0f}, Max: {dt_array.max():.0f}")
        print(f"  - P{percentile}: {dt_max:.0f}, #pairs: {len(all_dt)}")
        print(f"[INFO] Bin params:")
        print(f"  - Num bins: {num_bins}, Interval: {interval}")
        print(f"  - Max hist: {max_hist}, Sigma: {smooth_sigma:.2f}")
        
        # 4. 构建直方图
        hist = np.zeros((num_cameras, num_cameras, num_bins), dtype=np.float64)
        
        for pid, cams in pid_cam_data.items():
            cam_list = list(cams.keys())
            if len(cam_list) < 2:
                continue
            
            for i in range(len(cam_list)):
                for j in range(i + 1, len(cam_list)):
                    cam_i, cam_j = cam_list[i], cam_list[j]
                    for frame_i in cams[cam_i]:
                        for frame_j in cams[cam_j]:
                            dt = abs(frame_j - frame_i)
                            if dt <= 0:
                                continue
                            
                            if dt >= max_hist:
                                bin_idx = num_bins - 1
                            else:
                                bin_idx = int(dt // interval)
                            
                            hist[cam_i, cam_j, bin_idx] += 1.0
                            hist[cam_j, cam_i, bin_idx] += 1.0
        
        # 5. 高斯平滑和归一化
        sigma_bins = smooth_sigma / float(interval) if interval > 0 else 0.0
        
        def gauss1d_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
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
        
        for i in range(num_cameras):
            for j in range(num_cameras):
                if i == j:
                    continue
                s = float(hist[i, j].sum())
                if s > 0:
                    if sigma_bins > 0:
                        hist[i, j] = gauss1d_smooth(hist[i, j], sigma_bins)
                    hist[i, j] /= hist[i, j].sum() + 1e-12
                else:
                    # 未观测到的摄像头对：使用均匀分布，避免 st_prob=0 造成融合崩坏
                    hist[i, j] = 1.0 / float(num_bins)
        
        # 6. 计算可靠性矩阵
        reliability = self.compute_distribution_entropy(hist.astype(np.float32))
        
        # 7. 准备返回结果
        return {
            'distribution': hist.astype(np.float32),
            'reliability': reliability,
            'interval': interval,
            'max_hist': max_hist,
            'smooth_sigma': smooth_sigma,
            'num_bins': num_bins,
            'num_cameras': num_cameras,
            'dataset': dataset
        }


def main():
    parser = argparse.ArgumentParser(description='Build uncertainty-aware spatio-temporal histogram')
    parser.add_argument('--train-dir', type=str, required=True,
                       help='Path to training images directory')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=[
                           'market', 'market1501',
                           'duke', 'dukemtmc',
                           'occ', 'occluded', 'occluded_duke', 'occluded_dukemtmc',
                           'msmt', 'msmt17',
                           'veri', 'veri776',
                           'custom'
                       ],
                       help='Dataset name')
    parser.add_argument('--output-npz', type=str, required=True,
                       help='Output .npz file path')
    parser.add_argument('--num-bins', type=int, default=80,
                       help='Number of time bins (default: 80)')
    parser.add_argument('--smooth-ratio', type=float, default=0.5,
                       help='Smoothing sigma ratio (default: 0.5)')
    parser.add_argument('--percentile', type=float, default=99.9,
                       help='Percentile for time diff cutoff (default: 99.9)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.train_dir):
        print(f"Error: Training directory not found: {args.train_dir}")
        sys.exit(1)
    
    # 创建构建器
    builder = UncertaintyAwareSTBuilder()
    
    # 扫描目录
    print(f"Scanning {args.dataset} training directory: {args.train_dir}")
    samples, num_cameras = builder.scan_train_dir(args.train_dir, args.dataset)
    
    print(f"Found {len(samples)} samples, {num_cameras} cameras")
    
    # 构建直方图
    print("Building uncertainty-aware spatio-temporal histogram...")
    result = builder.build_uncertainty_aware_histogram(
        samples=samples,
        num_cameras=num_cameras,
        dataset=args.dataset,
        num_bins=args.num_bins,
        smooth_sigma_ratio=args.smooth_ratio,
        percentile=args.percentile
    )
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output_npz) or '.', exist_ok=True)
    np.savez_compressed(args.output_npz, **result)
    
    # 打印统计信息
    print(f"\n[SAVED] Uncertainty-aware ST histogram saved to: {args.output_npz}")
    
    reliability = result['reliability']
    print(f"[RELIABILITY STATS]")
    print(f"  Mean: {reliability.mean():.4f}")
    print(f"  Min: {reliability.min():.4f} (cam pair: {np.unravel_index(reliability.argmin(), reliability.shape)})")
    print(f"  Max: {reliability.max():.4f} (cam pair: {np.unravel_index(reliability.argmax(), reliability.shape)})")
    
    # 显示可靠性矩阵示例
    print(f"\nReliability matrix (first 5x5):")
    for i in range(min(5, num_cameras)):
        row = [f"{reliability[i, j]:.3f}" for j in range(min(5, num_cameras))]
        print(f"  Cam{i}: {row}")


if __name__ == '__main__':
    main()