# -*- coding: utf-8 -*-
"""
Enhanced Stage-2 Processor for CLIP-ReID

修复了所有错误，完全兼容原有接口
集成了多种不确定性感知ST融合方法
"""

import csv
import logging
import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from loss.supcontrast import SupConLoss

# ST-Meta fusion for training features
from model.stmeta import fuse_train_feats
# ST histogram utilities
from utils.st_histogram_fusion import load_distribution_npz, parse_frame_ids

# =====================================================================
# 0. 通用工具函数
# =====================================================================

def _cfg_get(cfg, dotted, default=None):
    """Safely get nested attribute from cfg by dotted path."""
    cur = cfg
    for k in dotted.split('.'):
        if not hasattr(cur, k):
            return default
        cur = getattr(cur, k)
    return cur

def _unpack_forward_output(out):
    """解包模型输出，适配不同实现"""
    score, feat, image_features = None, None, None
    if isinstance(out, dict):
        score = out.get("score")
        feat = out.get("feat") or out.get("feats") or out.get("features")
        image_features = out.get("image_features") or out.get("img_features")
    elif isinstance(out, (list, tuple)):
        if len(out) >= 3:
            score, feat, image_features = out[0], out[1], out[2]
        elif len(out) == 2:
            score, feat = out[0], out[1]
        elif len(out) == 1:
            feat = out[0]
        else:
            feat = out
    else:
        feat = out
    return score, feat, image_features

def _pick_image_features(feat, fallback_tensor=None):
    """若未显式给 image_features，则从 feat 中挑一个 2D tensor"""
    if fallback_tensor is not None:
        return fallback_tensor
    if isinstance(feat, (list, tuple)):
        for cand in (1, 0):
            if len(feat) > cand and torch.is_tensor(feat[cand]):
                t = feat[cand]
                if t.dim() == 2:
                    return t
        for x in feat:
            if torch.is_tensor(x) and x.dim() == 2:
                return x
        return None
    elif torch.is_tensor(feat):
        return feat if feat.dim() == 2 else None
    return None

# =====================================================================
# 1. 生物启发增强（BIO） - 修复版本
# =====================================================================

def optimized_bio_enhancement(q_feats, g_feats, sim_mat, base_strength=0.05, power=2.0, topk=5):
    """
    优化的生物启发增强：基于查询难度的自适应增强
    
    修复了数值计算问题
    """
    Q, D = q_feats.shape
    G = g_feats.shape[0]

    if G < 3:
        print(f"[Bio Warning] Gallery too small ({G}), skipping enhancement")
        return q_feats, torch.zeros(Q, device=q_feats.device), torch.zeros(Q, device=q_feats.device)

    k = min(topk, G)
    topk_sims, topk_idxs = torch.topk(sim_mat, k=k, dim=1)  # [Q,K]

    # 难度：低方差 + 低均值 -> 高难度
    sim_variance = topk_sims.var(dim=1)  # [Q]
    sim_mean = topk_sims.mean(dim=1)     # [Q]
    
    # 修复除零错误
    eps = 1e-8
    
    # 归一化方差
    sim_variance_min = sim_variance.min()
    sim_variance_max = sim_variance.max()
    if sim_variance_max - sim_variance_min < eps:
        norm_variance = torch.zeros_like(sim_variance)
    else:
        norm_variance = (sim_variance - sim_variance_min) / (sim_variance_max - sim_variance_min + eps)
    
    # 归一化均值
    sim_mean_min = sim_mean.min()
    sim_mean_max = sim_mean.max()
    if sim_mean_max - sim_mean_min < eps:
        norm_mean = torch.zeros_like(sim_mean)
    else:
        norm_mean = (sim_mean - sim_mean_min) / (sim_mean_max - sim_mean_min + eps)
    
    difficulty_scores = (1.0 - norm_variance) * (1.0 - norm_mean)
    difficulty_scores = difficulty_scores.clamp(min=0.1, max=0.9)

    # 强度：随难度非线性增长
    sigmoid_input = power * (difficulty_scores - 0.5)
    sigmoid_factor = torch.sigmoid(sigmoid_input)
    dynamic_base = base_strength * (0.7 + 0.6 * difficulty_scores)
    enhancement_strengths = dynamic_base * sigmoid_factor

    # 方向：topK 邻居残差的 softmax 加权平均
    weights = F.softmax(topk_sims / 0.05, dim=1)  # 温度更小 -> 权重更尖锐
    k_feats = g_feats[topk_idxs]                  # [Q,K,D]
    direction_vectors = k_feats - q_feats.unsqueeze(1)
    weighted_direction = torch.sum(direction_vectors * weights.unsqueeze(2), dim=1)
    weighted_direction = F.normalize(weighted_direction, p=2, dim=1)

    enhanced = q_feats + enhancement_strengths.unsqueeze(1) * weighted_direction
    enhanced = F.normalize(enhanced, p=2, dim=1)

    # 诊断信息
    feat_change = torch.norm(enhanced - q_feats, dim=1).mean().item()
    avg_strength = enhancement_strengths.mean().item()
    strong_count = (enhancement_strengths > (avg_strength * 1.5)).sum().item()

    print("\n" + "="*60)
    print("BIOLOGICALLY-INSPIRED ENHANCEMENT DIAGNOSTICS")
    print("="*60)
    print(f"• Query Analysis: Q={Q}, Gallery={G}, topK={k}")
    print(f"• Similarity Mean: {sim_mean.mean().item():.3f} "
          f"(Min: {sim_mean_min.item():.3f}, Max: {sim_mean_max.item():.3f})")
    print(f"• Similarity Variance Mean: {sim_variance.mean().item():.3f}")

    print(f"• Difficulty: Mean={difficulty_scores.mean().item():.3f}, "
          f"Min={difficulty_scores.min().item():.3f}, Max={difficulty_scores.max().item():.3f}")
    easy_count = (difficulty_scores < 0.3).sum().item()
    hard_count = (difficulty_scores > 0.7).sum().item()
    print(f"  - Easy(<0.3): {easy_count}/{Q} ({easy_count/Q*100:.1f}%), "
          f"Hard(>0.7): {hard_count}/{Q} ({hard_count/Q*100:.1f}%)")

    print(f"• Enhancement Strength: Mean={enhancement_strengths.mean().item():.4f}, "
          f"Min={enhancement_strengths.min().item():.4f}, Max={enhancement_strengths.max().item():.4f}")
    print(f"  - Strongly Enhanced: {strong_count}/{Q} ({strong_count/Q*100:.1f}%)")
    print(f"• Avg Feature Change: {feat_change:.6f}")
    if feat_change > 0.08:
        print("  ⚠️  WARNING: Feature change > 0.08, may disrupt subsequent AQE/ST")
    elif feat_change < 0.01:
        print("  ℹ️  INFO: Feature change < 0.01, enhancement may be too weak")
    else:
        print("  ✓ Feature change within safe range (0.01-0.08)")
    print("="*60 + "\n")

    return enhanced, enhancement_strengths, difficulty_scores

def minimal_bio_enhancement(q_feats, g_feats, sim_mat, strength=0.02, hard_sample_ratio=0.1):
    """最小化 BIO：只对最困难的样本进行增强"""
    Q, D = q_feats.shape
    mean_sims = sim_mat.mean(dim=1)  # [Q]

    threshold_idx = max(1, int(Q * hard_sample_ratio))
    threshold = torch.kthvalue(mean_sims, threshold_idx).values
    enhance_mask = mean_sims < threshold

    if not enhance_mask.any():
        return q_feats, torch.zeros(Q, device=q_feats.device)

    _, top1_idx = torch.topk(sim_mat, k=1, dim=1)
    direction = g_feats[top1_idx.squeeze()] - q_feats
    direction = F.normalize(direction, p=2, dim=1)

    enhanced = q_feats + enhance_mask.unsqueeze(1) * strength * direction
    enhanced = F.normalize(enhanced, p=2, dim=1)

    enhanced_count = enhance_mask.sum().item()
    print(f"[Minimal Bio] Enhanced {enhanced_count}/{Q} hardest queries ({enhanced_count/Q*100:.1f}%), strength={strength}")
    return enhanced, enhance_mask.float()

# =====================================================================
# 2. 多种不确定性感知ST融合方法（新增核心函数）
# =====================================================================

def uncertainty_aware_st_fusion(sim_vis, st_model, q_camids, g_camids, q_frames, g_frames,
                               lambda_st=0.16,
                               uncertainty_method='sqrt',
                               reliability_min=0.4,
                               reliability_floor=0.1,
                               lambda_boost=1.0,
                               sigmoid_k=5.0,
                               piecewise_thresholds=[0.4, 0.7],
                               piecewise_weights=[0.4, 0.7, 1.0]):
    """
    支持多种不确定性映射方法的ST融合
    
    参数说明：
    - uncertainty_method: 不确定性映射方法，可选值：
        'linear': 线性映射 λ = λ_st * max(R, reliability_min)
        'sqrt': 平方根映射 λ = λ_st * sqrt(max(R, reliability_min))
        'sigmoid': Sigmoid映射 λ = λ_st * sigmoid(k*(R-0.5))
        'piecewise': 分段函数映射
        'boosted_linear': 增强线性 λ = λ_st * max(R, reliability_min) * lambda_boost
        'fixed': 固定ST融合 λ = λ_st，不使用可靠性矩阵
    """
    device = sim_vis.device
    
    # 确保ST模型的张量在正确的设备上
    st_model.distribution = st_model.distribution.to(device)
    if hasattr(st_model, 'reliability') and st_model.reliability is not None:
        st_model.reliability = st_model.reliability.to(device)
    
    # 转换为张量
    q_cam_tensor = torch.from_numpy(q_camids).long().to(device)
    q_frame_tensor = torch.from_numpy(q_frames).long().to(device)
    g_cam_tensor = torch.from_numpy(g_camids).long().to(device)
    g_frame_tensor = torch.from_numpy(g_frames).long().to(device)
    
    # 计算时空概率
    time_diff = torch.abs(q_frame_tensor.unsqueeze(1) - g_frame_tensor.unsqueeze(0))
    bin_idx = (time_diff // st_model.interval).long()
    bin_idx = torch.clamp(bin_idx, 0, st_model.num_bins - 1)
    
    # 获取摄像头索引
    q_cam_exp = q_cam_tensor.unsqueeze(1).expand(-1, g_cam_tensor.size(0))
    g_cam_exp = g_cam_tensor.unsqueeze(0).expand(q_cam_tensor.size(0), -1)
    
    # 获取时空概率
    st_probs = st_model.distribution[q_cam_exp.long(), g_cam_exp.long(), bin_idx]
    
    # 获取可靠性矩阵
    if hasattr(st_model, 'reliability') and st_model.reliability is not None:
        reliability_matrix = st_model.reliability  # [C, C]
        pair_reliability = reliability_matrix[q_cam_exp.long(), g_cam_exp.long()]  # [Q, G]
        
        # 应用可靠性下限
        R = torch.clamp(pair_reliability, reliability_min, 1.0)
        
        # 根据方法计算自适应lambda
        if uncertainty_method in ['fixed', 'constant', 'none', 'static', 'fixed_st']:
            adaptive_lambda = torch.full_like(R, float(lambda_st))
        
        elif uncertainty_method == 'linear':
            adaptive_lambda = lambda_st * R
        
        elif uncertainty_method == 'sqrt':
            adaptive_lambda = lambda_st * torch.sqrt(R)
        
        elif uncertainty_method == 'sigmoid':
            # Sigmoid映射，使中间区域变化更敏感
            sigmoid_input = sigmoid_k * (R - 0.5)
            sigmoid_factor = torch.sigmoid(sigmoid_input)
            adaptive_lambda = lambda_st * sigmoid_factor
        
        elif uncertainty_method == 'piecewise':
            # 分段函数
            adaptive_lambda = torch.zeros_like(R)
            low_mask = R < piecewise_thresholds[0]
            medium_mask = (R >= piecewise_thresholds[0]) & (R < piecewise_thresholds[1])
            high_mask = R >= piecewise_thresholds[1]
            
            adaptive_lambda[low_mask] = lambda_st * piecewise_weights[0]
            adaptive_lambda[medium_mask] = lambda_st * piecewise_weights[1]
            adaptive_lambda[high_mask] = lambda_st * piecewise_weights[2]
        
        elif uncertainty_method == 'boosted_linear':
            adaptive_lambda = lambda_st * R * lambda_boost
        
        else:
            # 默认线性
            adaptive_lambda = lambda_st * R
        
        # 确保lambda在合理范围内
        adaptive_lambda = torch.clamp(adaptive_lambda, 0.0, 1.0)
        
        # 应用可靠性地板值（防止过度惩罚）
        adaptive_lambda = torch.max(adaptive_lambda, torch.tensor(reliability_floor * lambda_st, device=device))
        
        # 不确定性感知融合
        sim_fused = (1.0 - adaptive_lambda) * sim_vis + adaptive_lambda * st_probs
        
        # 计算自适应lambda统计
        lambda_stats = {
            'mean': adaptive_lambda.mean().item(),
            'std': adaptive_lambda.std().item(),
            'min': adaptive_lambda.min().item(),
            'max': adaptive_lambda.max().item(),
            'reliability_mean': R.mean().item(),
            'high_reliability_ratio': (R > 0.7).float().mean().item() * 100,
            'low_reliability_ratio': (R < 0.4).float().mean().item() * 100,
        }
        
        return sim_fused, lambda_stats, adaptive_lambda
    
    else:
        # 没有可靠性矩阵，使用静态融合
        sim_fused = (1.0 - lambda_st) * sim_vis + lambda_st * st_probs
        lambda_stats = {'mean': lambda_st, 'std': 0.0, 'min': lambda_st, 'max': lambda_st}
        return sim_fused, lambda_stats, None
def _normalize_device_list(devices, fallback="cuda"):
    if devices is None:
        if torch.cuda.is_available():
            return [fallback]
        return ["cpu"]

    if isinstance(devices, int):
        devices = [devices]

    if isinstance(devices, str):
        s = devices.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [x.strip().strip("'").strip('"') for x in s.split(",") if x.strip()]
        devices = parts

    if not isinstance(devices, (list, tuple)):
        devices = [devices]

    out = []
    for d in devices:
        if isinstance(d, int):
            out.append(f"cuda:{d}")
        elif isinstance(d, str):
            ds = d.strip()
            if ds == "cpu":
                out.append("cpu")
            elif ds.startswith("cuda:"):
                out.append(ds)
            elif ds == "cuda":
                out.append("cuda:0")
            elif ds.isdigit():
                out.append(f"cuda:{ds}")

    if not out:
        return [fallback] if torch.cuda.is_available() else ["cpu"]
    return out


def _resolve_blockwise_devices(cfg, fallback_device="cuda"):
    cfg_devices = _cfg_get(cfg, "ST.BLOCKWISE_DEVICES", None)
    if cfg_devices is None:
        if torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return ["cpu"]
    return _normalize_device_list(cfg_devices, fallback=fallback_device)


def _logistic_smoothing_score(x: torch.Tensor, lam: float = 1.0, gamma: float = 5.0) -> torch.Tensor:
    """
    st-ReID Logistic Smoothing:
        f(x; lambda, gamma) = 1 / (1 + lambda * exp(-gamma * x))
    Equivalent stable form:
        sigmoid(gamma * x - log(lambda))
    """
    lam = max(float(lam), 1e-12)
    return torch.sigmoid(float(gamma) * x.float() - np.log(lam))


def _is_ls_fusion_rule(fusion_rule: str) -> bool:
    return str(fusion_rule).lower() in {"st_reid_ls", "logistic_smoothing", "ls", "streid_ls"}


def _is_fixed_st_method(uncertainty_method: str) -> bool:
    return str(uncertainty_method).lower() in {"fixed", "constant", "none", "static", "fixed_st"}


def _compute_adaptive_lambda(
    R,
    lambda_st=0.13,
    uncertainty_method="sqrt",
    reliability_min=0.4,
    reliability_floor=0.1,
    lambda_boost=1.0,
    sigmoid_k=5.0,
    piecewise_thresholds=(0.4, 0.7),
    piecewise_weights=(0.4, 0.7, 1.0),
):
    """
    Compute pair-adaptive ST injection weights.

    Important variants:
    - fixed/constant/static: true Fixed-ST, lambda_qg = lambda_st, ignores R.
    - linear: UST-linear, lambda_qg = lambda_st * max(R, reliability_min).
    - sqrt: UST-sqrt, lambda_qg = lambda_st * sqrt(max(R, reliability_min)).
    """
    method = str(uncertainty_method).lower()

    if _is_fixed_st_method(method):
        adaptive_lambda = torch.full_like(R, float(lambda_st))
    else:
        R = torch.clamp(R, reliability_min, 1.0)

        if method == "linear":
            adaptive_lambda = lambda_st * R
        elif method == "sqrt":
            adaptive_lambda = lambda_st * torch.sqrt(R + 1e-8)
        elif method == "sigmoid":
            sigmoid_input = sigmoid_k * (R - 0.5)
            sigmoid_factor = torch.sigmoid(sigmoid_input)
            adaptive_lambda = lambda_st * sigmoid_factor
        elif method == "piecewise":
            adaptive_lambda = torch.zeros_like(R)
            low_mask = R < piecewise_thresholds[0]
            medium_mask = (R >= piecewise_thresholds[0]) & (R < piecewise_thresholds[1])
            high_mask = R >= piecewise_thresholds[1]
            adaptive_lambda[low_mask] = lambda_st * piecewise_weights[0]
            adaptive_lambda[medium_mask] = lambda_st * piecewise_weights[1]
            adaptive_lambda[high_mask] = lambda_st * piecewise_weights[2]
        elif method == "boosted_linear":
            adaptive_lambda = lambda_st * R * lambda_boost
        else:
            adaptive_lambda = lambda_st * R

    adaptive_lambda = torch.clamp(adaptive_lambda, 0.0, 1.0)
    floor_val = reliability_floor * lambda_st
    adaptive_lambda = torch.maximum(adaptive_lambda, torch.full_like(adaptive_lambda, floor_val))
    return adaptive_lambda


def fuse_st_blockwise(
    sim_vis,
    st_model,
    q_camids,
    g_camids,
    q_frames,
    g_frames,
    lambda_st=0.13,
    uncertainty_method="sqrt",
    reliability_min=0.4,
    reliability_floor=0.1,
    lambda_boost=1.0,
    sigmoid_k=5.0,
    piecewise_thresholds=(0.4, 0.7),
    piecewise_weights=(0.4, 0.7, 1.0),
    chunk_size=128,
    devices=None,
    return_lambda_stats=False,
    fusion_rule="weighted_sum",
    ls_lambda0=1.0,
    ls_gamma0=5.0,
    ls_lambda1=2.0,
    ls_gamma1=5.0,
):
    """
    OOM-safe ST fusion.

    Supported fusion rules:
    - weighted_sum: (1-lambda_qg) * S_vis + lambda_qg * p_st.
        * uncertainty_method=fixed  -> true Fixed-ST, lambda_qg=lambda_st.
        * uncertainty_method=linear -> UST-linear, lambda_qg=lambda_st*R.
        * uncertainty_method=sqrt   -> UST-sqrt, default R^2ID UST.
    - st_reid_ls: st-ReID Logistic Smoothing baseline,
        f(S_vis;lambda0,gamma0) * f(p_st;lambda1,gamma1).
    """
    if not torch.is_tensor(sim_vis):
        sim_vis = torch.as_tensor(sim_vis)

    Q, G = sim_vis.shape
    devices = _normalize_device_list(devices)
    sim_out = torch.empty((Q, G), dtype=torch.float16, device="cpu")

    g_camids = np.asarray(g_camids, dtype=np.int64)
    g_frames = np.asarray(g_frames, dtype=np.int64)
    q_camids = np.asarray(q_camids, dtype=np.int64)
    q_frames = np.asarray(q_frames, dtype=np.int64)

    lambda_sum = 0.0
    lambda_sq_sum = 0.0
    lambda_min = float("inf")
    lambda_max = float("-inf")
    lambda_count = 0

    use_ls = _is_ls_fusion_rule(fusion_rule)

    for block_id, start in enumerate(range(0, Q, chunk_size)):
        end = min(start + chunk_size, Q)
        dev = devices[block_id % len(devices)]

        sim_chunk = sim_vis[start:end]
        if dev == "cpu":
            if sim_chunk.device.type != "cpu":
                sim_chunk = sim_chunk.cpu()
        else:
            if str(sim_chunk.device) != dev:
                sim_chunk = sim_chunk.to(dev, non_blocking=True)

        q_cam_tensor = torch.as_tensor(q_camids[start:end], dtype=torch.long, device=dev)
        q_frame_tensor = torch.as_tensor(q_frames[start:end], dtype=torch.long, device=dev)
        g_cam_tensor = torch.as_tensor(g_camids, dtype=torch.long, device=dev)
        g_frame_tensor = torch.as_tensor(g_frames, dtype=torch.long, device=dev)

        distribution = st_model.distribution.to(dev)
        reliability = None
        if hasattr(st_model, "reliability") and st_model.reliability is not None:
            reliability = st_model.reliability.to(dev)

        time_diff = torch.abs(q_frame_tensor.unsqueeze(1) - g_frame_tensor.unsqueeze(0))
        bin_idx = torch.clamp((time_diff // st_model.interval).long(), 0, st_model.num_bins - 1)

        q_cam_exp = q_cam_tensor.unsqueeze(1).expand(-1, G)
        g_cam_exp = g_cam_tensor.unsqueeze(0).expand(end - start, -1)
        st_probs = distribution[q_cam_exp, g_cam_exp, bin_idx]

        pair_reliability = None
        adaptive_lambda = None

        if use_ls:
            sim_prob = _logistic_smoothing_score(sim_chunk, lam=ls_lambda0, gamma=ls_gamma0)
            st_prob = _logistic_smoothing_score(st_probs, lam=ls_lambda1, gamma=ls_gamma1)
            fused = sim_prob * st_prob
        else:
            if reliability is not None:
                pair_reliability = reliability[q_cam_exp, g_cam_exp]
                adaptive_lambda = _compute_adaptive_lambda(
                    pair_reliability,
                    lambda_st=lambda_st,
                    uncertainty_method=uncertainty_method,
                    reliability_min=reliability_min,
                    reliability_floor=reliability_floor,
                    lambda_boost=lambda_boost,
                    sigmoid_k=sigmoid_k,
                    piecewise_thresholds=piecewise_thresholds,
                    piecewise_weights=piecewise_weights,
                )
                fused = (1.0 - adaptive_lambda) * sim_chunk + adaptive_lambda * st_probs

                if return_lambda_stats:
                    lam = adaptive_lambda.detach().float()
                    lambda_sum += lam.sum().item()
                    lambda_sq_sum += (lam ** 2).sum().item()
                    lambda_min = min(lambda_min, lam.min().item())
                    lambda_max = max(lambda_max, lam.max().item())
                    lambda_count += lam.numel()
            else:
                fused = (1.0 - lambda_st) * sim_chunk + lambda_st * st_probs

        sim_out[start:end] = fused.detach().to("cpu", dtype=torch.float16)

        del sim_chunk, q_cam_tensor, q_frame_tensor, g_cam_tensor, g_frame_tensor
        del distribution, time_diff, bin_idx, q_cam_exp, g_cam_exp, st_probs, fused
        if reliability is not None:
            del reliability
        if pair_reliability is not None:
            del pair_reliability
        if adaptive_lambda is not None:
            del adaptive_lambda
        if dev != "cpu":
            torch.cuda.empty_cache()

    if return_lambda_stats:
        if lambda_count > 0:
            mean = lambda_sum / lambda_count
            var = max(lambda_sq_sum / lambda_count - mean * mean, 0.0)
            stats = {"mean": mean, "std": var ** 0.5, "min": lambda_min, "max": lambda_max}
        else:
            stats = None
        return sim_out, stats

    return sim_out

# =====================================================================
# 3. AQE & ST-guided AQE
# =====================================================================

def _alpha_to_tensor(alpha: Union[float, torch.Tensor], num_q: int, device: torch.device) -> torch.Tensor:
    """把 alpha 统一成 [Q,1] tensor"""
    if torch.is_tensor(alpha):
        a = alpha.to(device)
        if a.dim() == 0:
            a = a.expand(num_q)
        if a.dim() == 1:
            a = a.view(num_q, 1)
        elif a.dim() == 2 and a.shape[1] == 1:
            pass
        else:
            raise ValueError(f"alpha tensor shape not supported: {tuple(a.shape)}")
        return a
    return torch.full((num_q, 1), float(alpha), device=device)

def apply_aqe_gpu(q_feats, g_feats, sim_mat, topk=8, alpha: Union[float, torch.Tensor] = 1.0):
    """相似度加权的 AQE（支持 alpha 为标量或 [Q]）"""
    device = q_feats.device
    num_q, num_g = sim_mat.shape
    k = min(topk, num_g)
    if k <= 0:
        return q_feats

    topk_sims, topk_idxs = torch.topk(sim_mat, k=k, dim=1)  # [Q,K]
    weights = F.softmax(topk_sims / 0.1, dim=1)
    k_feats = g_feats[topk_idxs]                            # [Q,K,D]
    weighted_feat = torch.sum(k_feats * weights.unsqueeze(2), dim=1)

    a = _alpha_to_tensor(alpha, num_q, device)              # [Q,1]
    q_feats_new = q_feats + a * weighted_feat
    return F.normalize(q_feats_new, p=2, dim=1)

def apply_st_guided_aqe(q_feats, g_feats, sim_mat,
                        q_cams, q_frms, g_cams, g_frms,
                        st_model,
                        topk=5,
                        alpha: Union[float, torch.Tensor] = 0.8,
                        use_reliability: bool = False,
                        eta0: float = 0.5,
                        tau: float = 0.1):
    """ST 引导的 AQE（支持 per-query alpha，可选可靠性抬底）"""
    device = q_feats.device
    num_q, num_g = sim_mat.shape
    k = min(topk, num_g)
    if k <= 0:
        return q_feats

    topk_sims, indices = torch.topk(sim_mat, k=k, dim=1)  # [Q,K]

    q_cams = np.asarray(q_cams).astype(np.int64)
    g_cams = np.asarray(g_cams).astype(np.int64)
    q_frms = np.asarray(q_frms).astype(np.int64)
    g_frms = np.asarray(g_frms).astype(np.int64)

    q_c_exp = np.repeat(q_cams, k)
    q_f_exp = np.repeat(q_frms, k)
    flat_idxs = indices.detach().cpu().numpy().reshape(-1)
    g_c_neigh = g_cams[flat_idxs]
    g_f_neigh = g_frms[flat_idxs]

    hist = st_model.distribution.cpu().numpy()
    interval = int(st_model.interval)
    num_bins = int(st_model.num_bins)

    time_diff = np.abs(q_f_exp - g_f_neigh)
    bin_idxs = np.floor(time_diff / interval).astype(np.int64)
    bin_idxs = np.clip(bin_idxs, 0, num_bins - 1)

    st_probs_flat = hist[q_c_exp, g_c_neigh, bin_idxs]  # [Q*K]

    st_probs = torch.from_numpy(np.asarray(st_probs_flat, dtype=np.float32)).to(device).reshape(num_q, k)

    # 可靠性抬底（对不可靠 camera-pair，把 p_st 拉向均匀分布）
    if use_reliability and hasattr(st_model, 'reliability') and st_model.reliability is not None:
        R = st_model.reliability.cpu().numpy()  # np [C,C]
        eta_flat = eta0 * (1.0 - R[q_c_exp, g_c_neigh])  # [Q*K]
        eta = torch.from_numpy(eta_flat.astype(np.float32)).to(device).reshape(num_q, k)
        uniform = 1.0 / float(num_bins)
        st_probs = (1.0 - eta) * st_probs + eta * uniform

    joint = topk_sims * st_probs
    weights = F.softmax(joint / max(tau, 1e-6), dim=1)

    k_feats = g_feats[indices]
    weighted_feat = torch.sum(k_feats * weights.unsqueeze(2), dim=1)

    a = _alpha_to_tensor(alpha, num_q, device)
    q_feats_new = q_feats + a * weighted_feat
    return F.normalize(q_feats_new, p=2, dim=1)

# =====================================================================
# 4. OOM-safe 分批评测
# =====================================================================

def batched_evaluate(dist_mat, q_pids, g_pids, q_camids, g_camids, max_rank=50, batch_size=256):
    """OOM-safe 分批评测"""
    num_q, num_g = dist_mat.shape
    if num_g < max_rank:
        max_rank = num_g

    device = "cuda" if torch.cuda.is_available() else "cpu"

    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    cmc_accum = np.zeros(max_rank, dtype=np.float32)
    ap_sum = 0.0
    num_valid_q = 0.0

    for i in range(0, num_q, batch_size):
        batch_dist = dist_mat[i: i + batch_size].to(device)
        _, batch_indices = torch.sort(batch_dist, dim=1)
        batch_indices = batch_indices.cpu().numpy()

        for j in range(batch_indices.shape[0]):
            q_idx = i + j
            q_pid = q_pids[q_idx]
            q_cam = q_camids[q_idx]

            order = batch_indices[j]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_cam)
            keep = ~remove

            matches = (g_pids[order][keep] == q_pid)
            if not np.any(matches):
                continue

            cmc = matches.cumsum()
            cmc[cmc > 1] = 1
            cmc_accum += cmc[:max_rank]

            num_rel = matches.sum()
            tmp_cmc = matches.cumsum()
            tmp_cmc = tmp_cmc / np.arange(1, len(tmp_cmc) + 1)
            ap = (tmp_cmc * matches).sum() / num_rel

            ap_sum += ap
            num_valid_q += 1.0

    if num_valid_q > 0:
        all_cmc = cmc_accum / num_valid_q
        mAP = ap_sum / num_valid_q
    else:
        all_cmc = np.zeros(max_rank, dtype=np.float32)
        mAP = 0.0

    return all_cmc, mAP

# =====================================================================
# 4.5 轻量级 Re-ranking（k-reciprocal）工具
# ---------------------------------------------------------------------
# 说明：
# - Stage-2 主链路默认不包含 re-ranking；此处提供一个可选的、默认在 CPU 上执行的实现，
#   用于做“四版本”消融：VIS / VIS+RR / R²ID / R²ID+RR。
# - 该实现会在 CPU 上持有 (Q+G)x(Q+G) 距离矩阵（float16/float32），避免 GPU OOM。
# - 对于大图库（如 Duke: G≈1.6e4），计算与排序会较慢；建议先在小子集验证。
# =====================================================================

def cosine_distmat_blockwise(A: torch.Tensor,
                            B: torch.Tensor,
                            block: int = 1024,
                            out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """计算 1 - cosine_similarity 的距离矩阵（分块），返回 CPU Tensor。"""
    assert A.dim() == 2 and B.dim() == 2
    device = A.device if A.is_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    A_dev = A.to(device, non_blocking=True)
    B_dev = B.to(device, non_blocking=True)
    Bt = B_dev.t().contiguous()

    out = torch.empty((A_dev.size(0), B_dev.size(0)), device="cpu", dtype=out_dtype)

    use_amp = (device == "cuda")
    with torch.no_grad():
        for i in range(0, A_dev.size(0), block):
            Ai = A_dev[i:i + block]
            if use_amp:
                with amp.autocast():
                    sim = Ai @ Bt
            else:
                sim = Ai @ Bt
            out[i:i + block] = (1.0 - sim).to("cpu", dtype=out_dtype)

    return out


def _normalize_device_list(devices, fallback="cuda"):
    if devices is None:
        if torch.cuda.is_available():
            return [fallback]
        return ["cpu"]

    # 支持单个 int
    if isinstance(devices, int):
        devices = [devices]

    # 支持字符串: "0,1,2" / "[0,1,2]" / "cuda:0,cuda:1,cuda:2"
    if isinstance(devices, str):
        s = devices.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [x.strip().strip("'").strip('"') for x in s.split(",") if x.strip()]
        devices = parts

    # 兜底
    if not isinstance(devices, (list, tuple)):
        devices = [devices]

    out = []
    for d in devices:
        if isinstance(d, int):
            out.append(f"cuda:{d}")
        elif isinstance(d, str):
            ds = d.strip()
            if ds == "cpu":
                out.append("cpu")
            elif ds.startswith("cuda:"):
                out.append(ds)
            elif ds == "cuda":
                out.append("cuda:0")
            elif ds.isdigit():
                out.append(f"cuda:{ds}")
    return out or ([fallback] if torch.cuda.is_available() else ["cpu"])


def _resolve_blockwise_devices(cfg, fallback_device="cuda"):
    cfg_devices = _cfg_get(cfg, "ST.BLOCKWISE_DEVICES", None)

    # 如果没配，默认用所有可见 GPU
    if cfg_devices is None:
        if torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return ["cpu"]

    return _normalize_device_list(cfg_devices, fallback=fallback_device)

def final_dist_blockwise(
    q_feats: torch.Tensor,
    g_feats: torch.Tensor,
    q_camids, g_camids, q_frames, g_frames,
    st_model=None,
    lambda_st: float = 0.0,
    use_uncertainty_aware: bool = True,
    uncertainty_method: str = "sqrt",
    reliability_min: float = 0.4,
    reliability_floor: float = 0.1,
    lambda_boost: float = 1.0,
    sigmoid_k: float = 5.0,
    piecewise_thresholds=(0.4, 0.7),
    piecewise_weights=(0.4, 0.7, 1.0),
    chunk_size: int = 128,
    devices=None,
    out_dtype: torch.dtype = torch.float16,
    fusion_rule: str = "weighted_sum",
    ls_lambda0: float = 1.0,
    ls_gamma0: float = 5.0,
    ls_lambda1: float = 2.0,
    ls_gamma1: float = 5.0,
):
    """
    Directly compute final distance matrix blockwise.
    Supports both R^2ID weighted-sum ST fusion and st-ReID Logistic Smoothing.
    """
    assert q_feats.dim() == 2 and g_feats.dim() == 2
    Q, D = q_feats.shape
    G = g_feats.shape[0]

    devices = _normalize_device_list(devices, fallback="cuda")
    use_cuda = torch.cuda.is_available() and any(d != "cpu" for d in devices)
    if not use_cuda:
        devices = ["cpu"]

    q_cpu = q_feats.detach().cpu().float().contiguous()
    g_cpu = g_feats.detach().cpu().float().contiguous()

    q_camids = np.asarray(q_camids, dtype=np.int64)
    g_camids = np.asarray(g_camids, dtype=np.int64)
    q_frames = np.asarray(q_frames, dtype=np.int64)
    g_frames = np.asarray(g_frames, dtype=np.int64)

    out = torch.empty((Q, G), dtype=out_dtype, device="cpu")
    use_ls = _is_ls_fusion_rule(fusion_rule)

    per_dev = {}
    for dev in devices:
        g_feats_dev = g_cpu.to(dev, non_blocking=True)
        per_dev[dev] = {
            "g_feats": g_feats_dev,
            "g_feats_t": g_feats_dev.t().contiguous(),
            "g_cam": torch.as_tensor(g_camids, dtype=torch.long, device=dev),
            "g_frame": torch.as_tensor(g_frames, dtype=torch.long, device=dev),
        }
        if st_model is not None and (lambda_st > 0.0 or use_ls):
            per_dev[dev]["distribution"] = st_model.distribution.to(dev)
            if hasattr(st_model, "reliability") and st_model.reliability is not None:
                per_dev[dev]["reliability"] = st_model.reliability.to(dev)
            else:
                per_dev[dev]["reliability"] = None

    with torch.no_grad():
        for block_idx, start in enumerate(range(0, Q, chunk_size)):
            end = min(start + chunk_size, Q)
            dev = devices[block_idx % len(devices)]
            cache = per_dev[dev]

            q_chunk = q_cpu[start:end].to(dev, non_blocking=True)
            sim_chunk = q_chunk @ cache["g_feats_t"]

            if st_model is not None and (lambda_st > 0.0 or use_ls):
                q_cam = torch.as_tensor(q_camids[start:end], dtype=torch.long, device=dev)
                q_frame = torch.as_tensor(q_frames[start:end], dtype=torch.long, device=dev)

                time_diff = torch.abs(q_frame.unsqueeze(1) - cache["g_frame"].unsqueeze(0))
                bin_idx = torch.clamp(time_diff // int(st_model.interval), 0, int(st_model.num_bins) - 1)

                q_cam_exp = q_cam.unsqueeze(1).expand(-1, G)
                g_cam_exp = cache["g_cam"].unsqueeze(0).expand(end - start, -1)
                st_probs = cache["distribution"][q_cam_exp, g_cam_exp, bin_idx]

                if use_ls:
                    sim_prob = _logistic_smoothing_score(sim_chunk, lam=ls_lambda0, gamma=ls_gamma0)
                    st_prob = _logistic_smoothing_score(st_probs, lam=ls_lambda1, gamma=ls_gamma1)
                    fused = sim_prob * st_prob
                    dist_chunk = 1.0 - fused
                    del sim_prob, st_prob, fused
                elif use_uncertainty_aware and cache.get("reliability") is not None:
                    pair_R = cache["reliability"][q_cam_exp, g_cam_exp]
                    lam = _compute_adaptive_lambda(
                        pair_R,
                        lambda_st=lambda_st,
                        uncertainty_method=uncertainty_method,
                        reliability_min=reliability_min,
                        reliability_floor=reliability_floor,
                        lambda_boost=lambda_boost,
                        sigmoid_k=sigmoid_k,
                        piecewise_thresholds=piecewise_thresholds,
                        piecewise_weights=piecewise_weights,
                    )
                    fused = (1.0 - lam) * sim_chunk + lam * st_probs
                    dist_chunk = 1.0 - fused
                    del pair_R, lam, fused
                else:
                    fused = (1.0 - lambda_st) * sim_chunk + lambda_st * st_probs
                    dist_chunk = 1.0 - fused
                    del fused

                del q_cam, q_frame, time_diff, bin_idx, q_cam_exp, g_cam_exp, st_probs
            else:
                dist_chunk = 1.0 - sim_chunk

            out[start:end] = dist_chunk.detach().to("cpu", dtype=out_dtype)

            del q_chunk, sim_chunk, dist_chunk
            if dev != "cpu":
                torch.cuda.empty_cache()

    return out


def k_reciprocal_re_ranking(dist_qg: torch.Tensor,
                            dist_qq: torch.Tensor,
                            dist_gg: torch.Tensor,
                            k1: int = 20,
                            k2: int = 6,
                            lambda_value: float = 0.3) -> np.ndarray:
    """k-reciprocal re-ranking (Zhong et al., CVPR'17) 的 numpy 实现。"""
    qg = dist_qg.cpu().numpy().astype(np.float32, copy=False)
    qq = dist_qq.cpu().numpy().astype(np.float32, copy=False)
    gg = dist_gg.cpu().numpy().astype(np.float32, copy=False)

    query_num = qq.shape[0]
    gallery_num = gg.shape[0]
    all_num = query_num + gallery_num

    original_dist = np.zeros((all_num, all_num), dtype=np.float32)
    original_dist[:query_num, :query_num] = qq
    original_dist[:query_num, query_num:] = qg
    original_dist[query_num:, :query_num] = qg.T
    original_dist[query_num:, query_num:] = gg

    original_dist = np.maximum(original_dist, 0.0)
    original_dist = original_dist / (np.max(original_dist, axis=0, keepdims=True) + 1e-12)

    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)
    V = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(all_num):
        forward_k = initial_rank[i, :k1 + 1]
        backward_k = initial_rank[forward_k, :k1 + 1]
        fi = np.where(backward_k == i)[0]
        k_reciprocal_index = forward_k[fi]

        k_reciprocal_expansion_index = k_reciprocal_index.copy()
        for j in k_reciprocal_index:
            candidate_forward = initial_rank[j, :int(np.round(k1 / 2)) + 1]
            candidate_backward = initial_rank[candidate_forward, :int(np.round(k1 / 2)) + 1]
            fj = np.where(candidate_backward == j)[0]
            candidate_k_reciprocal = candidate_forward[fj]
            if len(np.intersect1d(candidate_k_reciprocal, k_reciprocal_index)) > (2. / 3) * len(candidate_k_reciprocal):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / (np.sum(weight) + 1e-12)

    if k2 > 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe

    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)
    for i in range(all_num):
        temp_min = np.zeros((1, all_num), dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        for ind in indNonZero:
            indImages.append(invIndex[ind])
        if len(indImages) > 0:
            indImages = np.unique(np.concatenate(indImages))
        else:
            indImages = np.array([], dtype=np.int32)

        for j in indImages:
            temp_min[0, j] += np.minimum(V[i, indNonZero], V[j, indNonZero]).sum()

        jaccard_dist[i] = 1 - temp_min / (2 - temp_min + 1e-12)

    final_dist = (1 - lambda_value) * jaccard_dist + lambda_value * original_dist
    return final_dist[:query_num, query_num:]

# =====================================================================
# 5. 数据集超参数配置（BIO + ST + AQE）
# =====================================================================

def get_st_aqe_hparams(cfg):
    """根据数据集名字返回ST & AQE超参数"""
    name = str(_cfg_get(cfg, "DATASETS.NAMES", "")).lower()

    h = dict(
        lambda_st=0.0,
        aqe_topk=8,
        aqe_alpha=1.0,
        use_st_guided=True,

        use_bio_enhance=True,
        bio_method="optimized",
        bio_base_strength=0.05,
        bio_power=2.0,
        bio_topk=5,
    )

    if "market" in name:
        h.update(dict(lambda_st=0.13, aqe_topk=8, aqe_alpha=1.0, use_st_guided=True,
                      bio_base_strength=0.5, bio_power=1.1, bio_topk=6))
    elif "occ" in name or "occluded" in name:
        h.update(dict(lambda_st=0.13, aqe_topk=8, aqe_alpha=1.0, use_st_guided=True,
                      bio_base_strength=0.5, bio_power=1.1, bio_topk=6))
    elif "duke" in name:
        h.update(dict(lambda_st=0.13, aqe_topk=8, aqe_alpha=1.0, use_st_guided=True,
                      bio_base_strength=0.5, bio_power=1.1, bio_topk=6))
    elif "msmt" in name:
        h.update(dict(lambda_st=0.13, aqe_topk=8, aqe_alpha=1.0, use_st_guided=True,
                      bio_base_strength=0.5, bio_power=1.1, bio_topk=6))
    elif "veri" in name:
        h.update(dict(lambda_st=0.13, aqe_topk=8, aqe_alpha=1.0, use_st_guided=True,
                      bio_base_strength=0.5, bio_power=1.1, bio_topk=6))

    # 允许YAML/命令行覆盖
    h["lambda_st"] = _cfg_get(cfg, "META.ST_HIST.LAMBDA", h["lambda_st"])
    h["aqe_topk"] = _cfg_get(cfg, "META.AQE.TOPK", h["aqe_topk"])
    h["aqe_alpha"] = _cfg_get(cfg, "META.AQE.ALPHA", h["aqe_alpha"])
    h["use_st_guided"] = _cfg_get(cfg, "META.AQE.ST_GUIDED", h["use_st_guided"])

    # BIO 覆盖
    h["use_bio_enhance"] = _cfg_get(cfg, "BIO.ENHANCE.ENABLE", h["use_bio_enhance"])
    h["bio_method"] = _cfg_get(cfg, "BIO.ENHANCE.METHOD", h["bio_method"])
    h["bio_base_strength"] = _cfg_get(cfg, "BIO.ENHANCE.BASE_STRENGTH", h["bio_base_strength"])
    h["bio_power"] = _cfg_get(cfg, "BIO.ENHANCE.POWER", h["bio_power"])
    h["bio_topk"] = _cfg_get(cfg, "BIO.ENHANCE.TOPK", h["bio_topk"])

    return h

# =====================================================================
# 6. 训练逻辑（Stage-2）
# =====================================================================

def do_train_stage2(cfg, model, center_criterion, train_loader_stage2, val_loader, optimizer, optimizer_center, scheduler, loss_fn, num_query, local_rank):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger("transreid.train")
    logger.info("Start training stage-2")

    if device == "cuda" and torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training (DataParallel).")
        model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    scaler = amp.GradScaler()

    # 预计算文本特征（若模型支持）
    num_classes = getattr(model.module if isinstance(model, nn.DataParallel) else model, "num_classes", None)
    text_features = None
    if num_classes is not None:
        batch_for_text = _cfg_get(cfg, "SOLVER.STAGE2.IMS_PER_BATCH", 64)
        iters = (num_classes + batch_for_text - 1) // batch_for_text
        tlist = []
        with torch.no_grad():
            for i in range(iters):
                label_list = torch.arange(i * batch_for_text,
                                          min((i + 1) * batch_for_text, num_classes),
                                          device=device)
                try:
                    with amp.autocast(enabled=True):
                        tf = model(label=label_list, get_text=True)
                    tlist.append(tf.detach().cpu())
                except Exception:
                    tlist = []
                    break
        if tlist:
            text_features = torch.cat(tlist, 0).to(device)

    max_epochs = _cfg_get(cfg, "SOLVER.STAGE2.MAX_EPOCHS", _cfg_get(cfg, "SOLVER.MAX_EPOCHS", 120))
    log_period = _cfg_get(cfg, "SOLVER.LOG_PERIOD", 50)
    eval_period = _cfg_get(cfg, "SOLVER.STAGE2.EVAL_PERIOD", _cfg_get(cfg, "SOLVER.EVAL_PERIOD", 0))

    for epoch in range(max_epochs):
        loss_meter.reset()
        acc_meter.reset()

        if scheduler is not None:
            try:
                scheduler.step()
            except TypeError:
                scheduler.step(epoch)

        model.train()

        for n_iter, batch in enumerate(train_loader_stage2):
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue
            img, vid, target_cam, target_view = batch[:4]

            optimizer.zero_grad(set_to_none=True)
            if optimizer_center is not None:
                optimizer_center.zero_grad(set_to_none=True)

            img = img.to(device, non_blocking=True)
            target = vid.to(device, non_blocking=True)

            if _cfg_get(cfg, "MODEL.SIE_CAMERA"):
                target_cam = target_cam.to(device, non_blocking=True)
            else:
                target_cam = None

            if _cfg_get(cfg, "MODEL.SIE_VIEW"):
                target_view = target_view.to(device, non_blocking=True)
            else:
                target_view = None

            with amp.autocast(enabled=True):
                out = model(x=img, label=target, cam_label=target_cam, view_label=target_view)
                score, feat, image_features = _unpack_forward_output(out)

                # 训练期ST融合
                if isinstance(feat, (list, tuple)) and len(feat) >= 2 and torch.is_tensor(feat[1]):
                    feat = fuse_train_feats(model, cfg, list(feat), camids=target_cam)
                elif torch.is_tensor(feat):
                    feat = fuse_train_feats(model, cfg, [feat, feat, feat], camids=target_cam)[1]

                # 文本对齐logits
                if text_features is not None:
                    im_feat = image_features if image_features is not None else _pick_image_features(feat)
                    if im_feat is not None and im_feat.dim() == 2:
                        logits = im_feat @ text_features.t()
                    else:
                        logits = None
                else:
                    logits = None

                # 组合损失
                try:
                    loss = loss_fn(score, feat, target, target_cam, logits)
                except TypeError:
                    loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # center loss
            metric_types = _cfg_get(cfg, "MODEL.METRIC_LOSS_TYPE", "")
            if isinstance(metric_types, (list, tuple)):
                has_center = any("center" in str(t).lower() for t in metric_types)
            else:
                has_center = "center" in str(metric_types).lower()

            if has_center and center_criterion is not None and optimizer_center is not None:
                for p in center_criterion.parameters():
                    if p.grad is not None:
                        p.grad.data *= (1.0 / _cfg_get(cfg, "SOLVER.CENTER_LOSS_WEIGHT", 0.0005))
                optimizer_center.step()

            # 统计训练acc
            with torch.no_grad():
                if logits is not None and torch.is_tensor(logits):
                    acc = (logits.max(1)[1] == target).float().mean()
                else:
                    s = score[0] if isinstance(score, list) else score
                    acc = (s.max(1)[1] == target).float().mean() if torch.is_tensor(s) else torch.tensor(0.0, device=device)

            loss_meter.update(loss.item(), img.size(0))
            acc_meter.update(acc, 1)

            if (n_iter + 1) % log_period == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Stage2 Epoch[{epoch}] Iter[{n_iter+1}/{len(train_loader_stage2)}] "
                    f"Loss: {loss_meter.avg:.3f}, Acc: {acc_meter.avg:.3f}, Lr: {lr:.2e}"
                )

        # 中途验证
        if eval_period and (epoch + 1) % eval_period == 0 and val_loader is not None:
            cmc1, cmc5 = do_inference(cfg, model, val_loader, num_query)
            logger.info(f"[Eval @ epoch {epoch+1}] Rank-1: {cmc1:.4f}, Rank-5: {cmc5:.4f}")

# =====================================================================
# 7. 推理逻辑（集成多种不确定性感知方法）
# =====================================================================

# =====================================================================
# 7. 推理逻辑（集成多种不确定性感知方法）- 修复版
# =====================================================================

# =====================================================================
# 7. 分桶实验与推理逻辑
# =====================================================================

def compute_query_ambiguity_scores(sim_mat: torch.Tensor, topk: int = 6) -> torch.Tensor:
    """
    根据论文中的定义计算每个 query 的歧义度 d_q:
        d_q = (1 - mu_hat_q) * (1 - var_hat_q)
    其中 mu_hat 和 var_hat 为当前 batch 内 min-max 归一化结果。
    """
    if sim_mat.dim() != 2:
        raise ValueError(f"sim_mat must be 2D, got shape={tuple(sim_mat.shape)}")

    num_q, num_g = sim_mat.shape
    if num_q == 0:
        return torch.empty(0, device=sim_mat.device)

    k = min(max(int(topk), 1), max(num_g, 1))
    if num_g == 0:
        return torch.zeros(num_q, device=sim_mat.device)

    topk_sims, _ = torch.topk(sim_mat, k=k, dim=1)
    sim_mean = topk_sims.mean(dim=1)
    sim_var = topk_sims.var(dim=1, unbiased=False)

    eps = 1e-8

    mean_min, mean_max = sim_mean.min(), sim_mean.max()
    if (mean_max - mean_min).abs() < eps:
        norm_mean = torch.zeros_like(sim_mean)
    else:
        norm_mean = (sim_mean - mean_min) / (mean_max - mean_min + eps)

    var_min, var_max = sim_var.min(), sim_var.max()
    if (var_max - var_min).abs() < eps:
        norm_var = torch.zeros_like(sim_var)
    else:
        norm_var = (sim_var - var_min) / (var_max - var_min + eps)

    dq = (1.0 - norm_mean) * (1.0 - norm_var)
    return dq.clamp(min=0.0, max=1.0)


def compute_per_query_metrics(dist_mat, q_pids, g_pids, q_camids, g_camids, max_rank=50, batch_size=256):
    """
    逐 query 计算 AP / Rank-1 / Rank-5。
    返回:
        {
            "valid": [Q] bool,
            "ap":    [Q] float,
            "rank1": [Q] float,
            "rank5": [Q] float,
            "first_hit_rank": [Q] int
        }
    """
    if not torch.is_tensor(dist_mat):
        dist_mat = torch.as_tensor(dist_mat)

    num_q, num_g = dist_mat.shape
    max_rank = min(max_rank, num_g) if num_g > 0 else 0

    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    valid = np.zeros(num_q, dtype=bool)
    ap = np.full(num_q, np.nan, dtype=np.float32)
    rank1 = np.full(num_q, np.nan, dtype=np.float32)
    rank5 = np.full(num_q, np.nan, dtype=np.float32)
    first_hit_rank = np.full(num_q, -1, dtype=np.int32)

    device = dist_mat.device
    with torch.no_grad():
        for i in range(0, num_q, batch_size):
            batch_dist = dist_mat[i:i + batch_size].to(device)
            _, batch_indices = torch.sort(batch_dist, dim=1)
            batch_indices = batch_indices.cpu().numpy()

            for j in range(batch_indices.shape[0]):
                q_idx = i + j
                q_pid = q_pids[q_idx]
                q_cam = q_camids[q_idx]

                order = batch_indices[j]
                remove = (g_pids[order] == q_pid) & (g_camids[order] == q_cam)
                keep = ~remove

                matches = (g_pids[order][keep] == q_pid)
                if not np.any(matches):
                    continue

                valid[q_idx] = True
                hit_positions = np.where(matches)[0]
                first_hit_rank[q_idx] = int(hit_positions[0] + 1)

                cmc = matches.cumsum()
                cmc[cmc > 1] = 1

                rank1[q_idx] = float(cmc[0]) if len(cmc) >= 1 else 0.0
                idx_r5 = min(4, len(cmc) - 1)
                rank5[q_idx] = float(cmc[idx_r5]) if idx_r5 >= 0 else 0.0

                num_rel = matches.sum()
                tmp_cmc = matches.cumsum().astype(np.float32)
                tmp_cmc = tmp_cmc / np.arange(1, len(tmp_cmc) + 1, dtype=np.float32)
                ap[q_idx] = float((tmp_cmc * matches).sum() / max(num_rel, 1))

    return {
        "valid": valid,
        "ap": ap,
        "rank1": rank1,
        "rank5": rank5,
        "first_hit_rank": first_hit_rank,
    }


def summarize_metric_subset(metric_dict, subset_mask):
    subset_mask = np.asarray(subset_mask, dtype=bool)
    joint_mask = subset_mask & metric_dict["valid"]
    n = int(joint_mask.sum())
    if n == 0:
        return {
            "count": 0,
            "mAP": float("nan"),
            "R1": float("nan"),
            "R5": float("nan"),
        }
    return {
        "count": n,
        "mAP": float(np.nanmean(metric_dict["ap"][joint_mask]) * 100.0),
        "R1": float(np.nanmean(metric_dict["rank1"][joint_mask]) * 100.0),
        "R5": float(np.nanmean(metric_dict["rank5"][joint_mask]) * 100.0),
    }


def build_tertile_groups(scores, valid_mask=None):
    """
    按样本数量三等分，而不是按数值区间均分。
    这样 low / medium / high 三组的样本量更稳定，更适合论文做 grouped experiment。
    """
    scores = np.asarray(scores, dtype=np.float32)
    finite_mask = np.isfinite(scores)
    if valid_mask is None:
        valid_mask = finite_mask
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & finite_mask

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        empty = np.zeros_like(scores, dtype=bool)
        return {
            "thresholds": (float("nan"), float("nan")),
            "groups": {"low": empty.copy(), "medium": empty.copy(), "high": empty.copy()},
        }

    sorted_idx = valid_idx[np.argsort(scores[valid_idx], kind="mergesort")]
    n = len(sorted_idx)
    split1 = max(1, n // 3) if n >= 3 else max(1, n)
    split2 = max(split1 + 1, (2 * n) // 3) if n >= 3 else n
    split2 = min(split2, n)

    low_idx = sorted_idx[:split1]
    med_idx = sorted_idx[split1:split2]
    high_idx = sorted_idx[split2:]

    # 极小数据时，保证 high 至少不为空
    if len(high_idx) == 0 and len(med_idx) > 0:
        high_idx = med_idx[-1:]
        med_idx = med_idx[:-1]

    low = np.zeros_like(scores, dtype=bool)
    medium = np.zeros_like(scores, dtype=bool)
    high = np.zeros_like(scores, dtype=bool)

    low[low_idx] = True
    medium[med_idx] = True
    high[high_idx] = True

    q1 = float(scores[low_idx[-1]]) if len(low_idx) > 0 else float("nan")
    if len(med_idx) > 0:
        q2 = float(scores[med_idx[-1]])
    elif len(high_idx) > 0:
        q2 = float(scores[high_idx[0]])
    else:
        q2 = float("nan")

    return {
        "thresholds": (q1, q2),
        "groups": {"low": low, "medium": medium, "high": high},
    }


def compute_query_reliability_scores(q_pids, g_pids, q_camids, g_camids, reliability_matrix):
    """
    为每个 query 计算一个 query-level camera-pair reliability 分数。
    这里采用:
        对该 query 的所有 cross-camera 正样本 gallery，
        取 R(q_cam, g_cam_pos) 的平均值。
    这样得到的是“该 query 可利用时空先验质量”的直接近似。
    """
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    if torch.is_tensor(reliability_matrix):
        rel = reliability_matrix.detach().cpu().numpy().astype(np.float32)
    else:
        rel = np.asarray(reliability_matrix, dtype=np.float32)

    scores = np.full(len(q_pids), np.nan, dtype=np.float32)

    for i in range(len(q_pids)):
        pos_mask = (g_pids == q_pids[i]) & (g_camids != q_camids[i])
        cams = g_camids[pos_mask]

        # 极少数情况下若 cross-camera 正样本为空，再退回到同 ID 全部正样本
        if cams.size == 0:
            pos_mask = (g_pids == q_pids[i])
            cams = g_camids[pos_mask]

        if cams.size == 0:
            continue

        vals = rel[int(q_camids[i]), cams.astype(np.int64)]
        if vals.size > 0:
            scores[i] = float(np.mean(vals))

    return scores


def log_grouped_table(logger, title, score_name, thresholds, rows):
    logger.info(f"\n{title}")
    logger.info(f"{score_name} thresholds (33% / 67%): q1={thresholds[0]:.4f}, q2={thresholds[1]:.4f}")
    logger.info("\n" + "-" * 92)
    logger.info(f"{'Method':<20} {'Group':<12} {'#Q':>8} {'mAP':>10} {'R1':>10}")
    logger.info("-" * 92)

    for row in rows:
        map_str = "nan" if not np.isfinite(row["mAP"]) else f"{row['mAP']:.2f}"
        r1_str = "nan" if not np.isfinite(row["R1"]) else f"{row['R1']:.2f}"
        logger.info(f"{row['method']:<20} {row['group']:<12} {row['count']:>8d} {map_str:>10} {r1_str:>10}")

    logger.info("-" * 92)


def save_grouped_rows_csv(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["experiment", "score_name", "threshold_q1", "threshold_q2", "method", "group", "count", "mAP", "R1", "R5"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_per_query_analysis_csv(csv_path, records):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _metric_dict_to_overall(metric_dict):
    valid = metric_dict["valid"]
    if valid.sum() == 0:
        return 0.0, np.zeros(50, dtype=np.float32)
    map_v = float(np.nanmean(metric_dict["ap"][valid]))
    r1 = float(np.nanmean(metric_dict["rank1"][valid]))
    r5 = float(np.nanmean(metric_dict["rank5"][valid]))
    cmc = np.zeros(5, dtype=np.float32)
    cmc[0] = r1
    cmc[4] = r5
    return map_v, cmc

def _metric_dict_to_map_r1_r5(metric_dict):
    """Return overall mAP / Rank-1 / Rank-5 from a per-query metric dictionary."""
    valid = metric_dict.get("valid")
    if valid is None or valid.sum() == 0:
        return 0.0, 0.0, 0.0
    mAP = float(np.nanmean(metric_dict["ap"][valid]))
    r1 = float(np.nanmean(metric_dict["rank1"][valid]))
    r5 = float(np.nanmean(metric_dict["rank5"][valid]))
    return mAP, r1, r5


def _metric_delta_str(value):
    return f"{value * 100:+.2f}%"


def do_inference(cfg, model, val_loader, num_query):
    """集成 BIO + UST + STG-AQE，并附带 d_q / R 分桶实验。"""
    logger = logging.getLogger("transreid.test")
    logger.info("Enter Reliability-Calibrated ST-ReID Inference")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # 0) grouped experiment 开关
    # -------------------------
    run_grouped_analysis = bool(_cfg_get(cfg, "ANALYSIS.GROUPED_EXPERIMENT.ENABLE", True))
    save_grouped_analysis = bool(_cfg_get(cfg, "ANALYSIS.GROUPED_EXPERIMENT.SAVE", True))
    grouped_dir = os.path.join(str(_cfg_get(cfg, "OUTPUT_DIR", ".")), "grouped_analysis")

    # -------------------------
    # 1) 初始化 ST
    # -------------------------
    # 不强制要求你改 yml：只要 META.ST_HIST.NPZ_PATH 存在，就自动启用 ST。
    npz_path = _cfg_get(cfg, "META.ST_HIST.NPZ_PATH", "")
    use_st = bool(_cfg_get(cfg, "META.ST_HIST.ENABLE", False)) or bool(npz_path)
    st_model = None
    if use_st:
        if npz_path:
            try:
                st_model = load_distribution_npz(npz_path, device=device)
                logger.info(f"ST model loaded from {npz_path}")
                logger.info(f"  distribution shape: {tuple(st_model.distribution.shape)}")
                logger.info(f"  interval={st_model.interval}, num_bins={st_model.num_bins}, num_cameras={st_model.num_cameras}")
                if hasattr(st_model, "reliability") and st_model.reliability is not None:
                    logger.info(
                        f"  reliability: mean={st_model.reliability.mean().item():.4f}, "
                        f"min={st_model.reliability.min().item():.4f}, "
                        f"max={st_model.reliability.max().item():.4f}"
                    )
            except Exception as e:
                logger.error(f"Failed to load ST model: {e}")
                use_st = False
                st_model = None
        else:
            logger.info("META.ST_HIST.ENABLE=True but NPZ_PATH empty, skip ST.")
            use_st = False

    # -------------------------
    # 2) 读取超参与 grouped experiment 配置
    # -------------------------
    hparams = get_st_aqe_hparams(cfg)
    lambda_st = float(hparams["lambda_st"])
    aqe_topk = int(hparams["aqe_topk"])
    aqe_alpha = float(hparams["aqe_alpha"])
    use_st_guided_aqe = bool(hparams["use_st_guided"] and use_st and st_model is not None)

    use_bio_enhance = bool(hparams["use_bio_enhance"])
    bio_method = str(hparams["bio_method"])
    bio_base_strength = float(hparams["bio_base_strength"])
    bio_power = float(hparams["bio_power"])
    bio_topk = int(hparams["bio_topk"])

    use_uncertainty_aware = bool(_cfg_get(cfg, "ST.UNCERTAINTY_AWARE", True))
    uncertainty_method = str(_cfg_get(cfg, "ST.UNCERTAINTY_METHOD", "sqrt")).lower()
    # 默认直接跑完整对照，不需要在 yml 里额外增加字段。
    # 最终主方法默认仍是 UST-sqrt + STG-AQE。
    fusion_rule = str(_cfg_get(cfg, "ST.FUSION_RULE", "weighted_sum")).lower()
    run_ls_baseline = bool(_cfg_get(cfg, "ST.RUN_LS_BASELINE", False))
    auto_run_fusion_variants = bool(_cfg_get(cfg, "ST.AUTO_RUN_FUSION_VARIANTS", True))
    run_linear_baseline = bool(_cfg_get(cfg, "ST.RUN_LINEAR_BASELINE", auto_run_fusion_variants))
    run_detailed_ablation = bool(_cfg_get(cfg, "ST.RUN_DETAILED_ABLATION", True))
    run_nobio_stgaqe_ablation = bool(_cfg_get(cfg, "ST.RUN_NOBIO_STGAQE_ABLATION", True))
    reliability_min = float(_cfg_get(cfg, "ST.RELIABILITY_MIN", 0.4))
    reliability_floor = float(_cfg_get(cfg, "ST.RELIABILITY_FLOOR", 0.1))
    lambda_boost = float(_cfg_get(cfg, "ST.LAMBDA_BOOST", 1.0))
    sigmoid_k = float(_cfg_get(cfg, "ST.SIGMOID_K", 5.0))
    piecewise_thresholds = _cfg_get(cfg, "ST.PIECEWISE_THRESHOLDS", [0.4, 0.7])
    piecewise_weights = _cfg_get(cfg, "ST.PIECEWISE_WEIGHTS", [0.4, 0.7, 1.0])
    ls_lambda0 = float(_cfg_get(cfg, "ST.LS_LAMBDA0", 1.0))
    ls_gamma0 = float(_cfg_get(cfg, "ST.LS_GAMMA0", 5.0))
    ls_lambda1 = float(_cfg_get(cfg, "ST.LS_LAMBDA1", 2.0))
    ls_gamma1 = float(_cfg_get(cfg, "ST.LS_GAMMA1", 5.0))

    logger.info("\n" + "=" * 84)
    logger.info("RELIABILITY-CALIBRATED INFERENCE CONFIG")
    logger.info("=" * 84)
    logger.info(f"BIO: enable={use_bio_enhance}, method={bio_method}, base_strength={bio_base_strength}, power={bio_power}, topk={bio_topk}")
    logger.info(f"ST : enable={use_st}, lambda_st={lambda_st:.4f}, fusion_rule={fusion_rule}, uncertainty_aware={use_uncertainty_aware}, method={uncertainty_method}")
    logger.info(f"Auto variants: fixed=True, linear={run_linear_baseline}, st-ReID-LS={run_ls_baseline}, detailed={run_detailed_ablation}, noBIO-STG-AQE={run_nobio_stgaqe_ablation}; final default={fusion_rule}/{uncertainty_method}")
    logger.info(f"ST-LS: run_baseline={run_ls_baseline}, lambda0={ls_lambda0}, gamma0={ls_gamma0}, lambda1={ls_lambda1}, gamma1={ls_gamma1}")
    logger.info(f"AQE: topk={aqe_topk}, alpha={aqe_alpha}, st_guided={use_st_guided_aqe}")
    logger.info(f"Grouped experiment: enable={run_grouped_analysis}, save={save_grouped_analysis}")
    logger.info("=" * 84 + "\n")

    model.eval()
    model.to(device)

    # -------------------------
    # 3) 提取所有特征
    # -------------------------
    feats, pids, camids, frames = [], [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if len(batch) >= 6:
                img, pid, camid, cam_batch, view_batch, path = batch[:6]
            elif len(batch) >= 4:
                img, pid, camid, path = batch[:4]
                cam_batch, view_batch = camid, None
            else:
                raise RuntimeError(f"Unexpected batch length {len(batch)}; expected >= 4.")

            img = img.to(device)
            input_kwargs = {}

            if _cfg_get(cfg, "MODEL.SIE_CAMERA"):
                input_kwargs["cam_label"] = cam_batch.to(device) if isinstance(cam_batch, torch.Tensor) else torch.tensor(cam_batch, device=device)

            if _cfg_get(cfg, "MODEL.SIE_VIEW") and view_batch is not None:
                input_kwargs["view_label"] = view_batch.to(device) if isinstance(view_batch, torch.Tensor) else torch.tensor(view_batch, device=device)

            try:
                out = model(img, **input_kwargs)
            except TypeError:
                out = model(img)

            if isinstance(out, dict):
                feat = out.get("feat") or out.get("features") or out.get("feats")
            elif isinstance(out, (list, tuple)):
                feat = out[-1]
            else:
                feat = out

            if isinstance(feat, (list, tuple)):
                picked = None
                for f in feat:
                    if torch.is_tensor(f) and f.dim() == 2:
                        picked = f
                        break
                feat = picked if picked is not None else feat[0]

            if feat is None:
                logger.warning(f"Batch {batch_idx}: no valid feature extracted, skip.")
                continue

            feats.append(feat.cpu())

            if isinstance(pid, torch.Tensor):
                pids.extend(pid.cpu().numpy().astype(np.int64).tolist())
            elif isinstance(pid, np.ndarray):
                pids.extend(pid.astype(np.int64).tolist())
            elif isinstance(pid, (list, tuple)):
                pids.extend([int(x) for x in pid])
            else:
                pids.append(int(pid))

            if isinstance(camid, torch.Tensor):
                camids.extend(camid.cpu().numpy().astype(np.int64).tolist())
            elif isinstance(camid, np.ndarray):
                camids.extend(camid.astype(np.int64).tolist())
            elif isinstance(camid, (list, tuple)):
                camids.extend([int(x) for x in camid])
            else:
                camids.append(int(camid))

            if use_st and path is not None:
                try:
                    if isinstance(path, str):
                        path_list = [path]
                    elif isinstance(path, (list, tuple)):
                        path_list = list(path)
                    else:
                        path_list = []
                    if path_list:
                        frm = parse_frame_ids(path_list)
                        frames.extend(np.asarray(frm, dtype=np.int64).tolist())
                    else:
                        frames.extend([0] * img.shape[0])
                except Exception as e:
                    logger.warning(f"Batch {batch_idx}: parse_frame_ids failed: {e}")
                    frames.extend([0] * img.shape[0])
            elif use_st:
                frames.extend([0] * img.shape[0])

    if not feats:
        raise RuntimeError("No features extracted during inference.")

    feats = torch.cat(feats, dim=0).to(device)
    feats = F.normalize(feats, p=2, dim=1)

    pids = np.asarray(pids, dtype=np.int64)
    camids = np.asarray(camids, dtype=np.int64)
    frames = np.asarray(frames, dtype=np.int64) if use_st else np.zeros_like(camids)

    qf = feats[:num_query]
    gf = feats[num_query:]

    q_pids, g_pids = pids[:num_query], pids[num_query:]
    q_camids, g_camids = camids[:num_query], camids[num_query:]
    q_frames, g_frames = frames[:num_query], frames[num_query:]

    qf_original = qf.clone()

    # -------------------------
    # 4) 先计算原始 dq，并执行 BIO
    # -------------------------
    sim_original = torch.mm(qf_original, gf.t())
    dq_scores = compute_query_ambiguity_scores(sim_original, topk=bio_topk)
    difficulty_scores = dq_scores.clone()

    bio_enhanced = False
    bio_diagnostics = {}

    print("\n" + "=" * 84)
    print("PHASE 1: BIO QUERY CORRECTION")
    print("=" * 84)

    if use_bio_enhance and bio_base_strength > 0:
        if bio_method == "optimized":
            qf_bio, enhancement_strengths, difficulty_scores = optimized_bio_enhancement(
                qf_original, gf, sim_original,
                base_strength=bio_base_strength,
                power=bio_power,
                topk=bio_topk
            )
            qf = qf_bio
            dq_scores = difficulty_scores.detach().clone()
            bio_diagnostics["avg_strength"] = float(enhancement_strengths.mean().item())
            bio_diagnostics["avg_dq"] = float(difficulty_scores.mean().item())
            bio_diagnostics["hard_ratio"] = float((difficulty_scores > 0.7).float().mean().item() * 100.0)
        elif bio_method == "minimal":
            qf_bio, enhance_mask = minimal_bio_enhancement(
                qf_original, gf, sim_original, strength=bio_base_strength, hard_sample_ratio=0.1
            )
            qf = qf_bio
            bio_diagnostics["enhanced_ratio"] = float(enhance_mask.mean().item() * 100.0)
            bio_diagnostics["avg_dq"] = float(dq_scores.mean().item())
        else:
            print(f"[BIO] unknown bio_method={bio_method}, skip BIO.")
            qf = qf_original

        bio_enhanced = not torch.equal(qf, qf_original)
        feat_change = torch.norm(qf - qf_original, dim=1).mean().item()
        bio_diagnostics["avg_feature_change"] = float(feat_change)
        print(f"[BIO] avg dq={dq_scores.mean().item():.4f}, avg feature change={feat_change:.6f}")
    else:
        print("[BIO] skipped.")
        qf = qf_original

    print("=" * 84 + "\n")

    # -------------------------
    # 5) BIO 后视觉结果
    # -------------------------
    sim_vis = torch.mm(qf, gf.t())
    dist_original = 1.0 - sim_original
    dist_vis = 1.0 - sim_vis

    metrics_original = compute_per_query_metrics(dist_original, q_pids, g_pids, q_camids, g_camids)
    metrics_vis = compute_per_query_metrics(dist_vis, q_pids, g_pids, q_camids, g_camids)

    mAP_original = float(np.nanmean(metrics_original["ap"][metrics_original["valid"]])) if metrics_original["valid"].any() else 0.0
    cmc_original = np.zeros(5, dtype=np.float32)
    cmc_original[0] = float(np.nanmean(metrics_original["rank1"][metrics_original["valid"]])) if metrics_original["valid"].any() else 0.0
    cmc_original[4] = float(np.nanmean(metrics_original["rank5"][metrics_original["valid"]])) if metrics_original["valid"].any() else 0.0

    mAP_vis = float(np.nanmean(metrics_vis["ap"][metrics_vis["valid"]])) if metrics_vis["valid"].any() else 0.0
    cmc_vis = np.zeros(5, dtype=np.float32)
    cmc_vis[0] = float(np.nanmean(metrics_vis["rank1"][metrics_vis["valid"]])) if metrics_vis["valid"].any() else 0.0
    cmc_vis[4] = float(np.nanmean(metrics_vis["rank5"][metrics_vis["valid"]])) if metrics_vis["valid"].any() else 0.0

    logger.info(f"[1] Original VIS : mAP={mAP_original:.2%}, Rank-1={cmc_original[0]:.2%}")
    logger.info(f"[2] BIO + VIS    : mAP={mAP_vis:.2%}, Rank-1={cmc_vis[0]:.2%}")

    # -------------------------
    # 6) Detailed ST ablation: isolate UST with BIO/AQE disabled and enabled
    # -------------------------
    lambda_stats = None
    fusion_type = "None"

    # Default metric handles. These are used later by grouped analysis and final summary.
    metrics_nobio_fixed_st = None
    metrics_nobio_linear_st = None
    metrics_nobio_sqrt_st = None
    metrics_nobio_ls_st = None
    metrics_fixed_st = metrics_vis
    metrics_linear_st = None
    metrics_bio_sqrt_st = None
    metrics_ls_st = None
    metrics_ust = metrics_vis
    metrics_bio_visual_aqe = None
    metrics_nobio_stgaqe = None

    sim_nobio_sqrt_st_for_aqe = None
    sim_ust = sim_vis.cpu().to(torch.float16)

    def _eval_sim_matrix(sim_matrix):
        return compute_per_query_metrics(1.0 - sim_matrix.float(), q_pids, g_pids, q_camids, g_camids)

    def _log_perf(tag, method_name, metric_dict, ref_metric=None):
        m, r1, r5 = _metric_dict_to_map_r1_r5(metric_dict)
        if ref_metric is not None:
            rm, rr1, _ = _metric_dict_to_map_r1_r5(ref_metric)
            logger.info(
                f"{tag:<8} {method_name:<34}: mAP={m:.2%}, Rank-1={r1:.2%} "
                f"(ΔmAP={m-rm:+.2%}, ΔR1={r1-rr1:+.2%})"
            )
        else:
            logger.info(f"{tag:<8} {method_name:<34}: mAP={m:.2%}, Rank-1={r1:.2%}")
        return m, r1, r5

    def _fuse_and_eval(sim_base, method="sqrt", fusion_rule_override="weighted_sum", return_stats=False):
        kwargs = dict(
            sim_vis=sim_base,
            st_model=st_model,
            q_camids=q_camids,
            g_camids=g_camids,
            q_frames=q_frames,
            g_frames=g_frames,
            lambda_st=lambda_st,
            chunk_size=chunk_size,
            devices=block_devices,
            return_lambda_stats=return_stats,
            fusion_rule=fusion_rule_override,
            ls_lambda0=ls_lambda0,
            ls_gamma0=ls_gamma0,
            ls_lambda1=ls_lambda1,
            ls_gamma1=ls_gamma1,
        )
        if fusion_rule_override == "weighted_sum":
            kwargs.update(dict(
                uncertainty_method=method,
                reliability_min=reliability_min,
                reliability_floor=reliability_floor,
                lambda_boost=lambda_boost,
                sigmoid_k=sigmoid_k,
                piecewise_thresholds=piecewise_thresholds,
                piecewise_weights=piecewise_weights,
            ))
        fused = fuse_st_blockwise(**kwargs)
        stats = None
        if return_stats:
            fused, stats = fused
        return fused, _eval_sim_matrix(fused), stats

    logger.info("\n" + "=" * 92)
    logger.info("DETAILED ABLATION: isolating UST under different module switches")
    logger.info("=" * 92)
    _log_perf("[0]", "Original VIS (no BIO / no ST / no AQE)", metrics_original)
    _log_perf("[1]", "BIO + VIS (BIO only)", metrics_vis, ref_metric=metrics_original)

    query_reliability_scores = None

    if use_st and st_model is not None and (lambda_st > 0.0 or _is_ls_fusion_rule(fusion_rule)):
        chunk_size = int(_cfg_get(cfg, "ST.BLOCKWISE_CHUNK_SIZE", 128))
        block_devices = _resolve_blockwise_devices(cfg, fallback_device=device)

        logger.info(f"ST blockwise: chunk_size={chunk_size}, devices={block_devices}")

        if hasattr(st_model, "reliability") and st_model.reliability is not None:
            query_reliability_scores = compute_query_reliability_scores(
                q_pids, g_pids, q_camids, g_camids, st_model.reliability
            )

        # ------------------------------------------------------------------
        # A. ST-only ablation on original visual similarity: BIO OFF, STG-AQE OFF.
        #    This directly answers whether UST works by itself.
        # ------------------------------------------------------------------
        if run_detailed_ablation:
            sim_nobio_fixed_st, metrics_nobio_fixed_st, _ = _fuse_and_eval(sim_original, method="fixed")
            _log_perf("[0a]", "VIS + Fixed-ST (no BIO / no AQE)", metrics_nobio_fixed_st, ref_metric=metrics_original)

            if run_linear_baseline and hasattr(st_model, "reliability") and st_model.reliability is not None:
                sim_nobio_linear_st, metrics_nobio_linear_st, _ = _fuse_and_eval(sim_original, method="linear")
                _log_perf("[0b]", "VIS + UST-linear (no BIO / no AQE)", metrics_nobio_linear_st, ref_metric=metrics_nobio_fixed_st)
                del sim_nobio_linear_st

            if hasattr(st_model, "reliability") and st_model.reliability is not None:
                sim_nobio_sqrt_st_for_aqe, metrics_nobio_sqrt_st, _ = _fuse_and_eval(sim_original, method="sqrt")
                _log_perf("[0c]", "VIS + UST-sqrt (no BIO / no AQE)", metrics_nobio_sqrt_st, ref_metric=metrics_nobio_fixed_st)

            if run_ls_baseline:
                sim_nobio_ls_st, metrics_nobio_ls_st, _ = _fuse_and_eval(sim_original, fusion_rule_override="st_reid_ls")
                _log_perf("[0d]", "VIS + st-ReID-LS naive (debug)", metrics_nobio_ls_st, ref_metric=metrics_original)
                del sim_nobio_ls_st

            # Free fixed matrix; keep sqrt only if needed for noBIO STG-AQE.
            del sim_nobio_fixed_st

        # ------------------------------------------------------------------
        # B. BIO + ST ablation: this is your relation-level calibration after BIO.
        # ------------------------------------------------------------------
        sim_fixed_st, metrics_fixed_st, _ = _fuse_and_eval(sim_vis, method="fixed")
        _log_perf("[2]", "BIO + Fixed-ST (no AQE)", metrics_fixed_st, ref_metric=metrics_vis)

        if run_linear_baseline and hasattr(st_model, "reliability") and st_model.reliability is not None:
            sim_linear_st, metrics_linear_st, _ = _fuse_and_eval(sim_vis, method="linear")
            _log_perf("[2a]", "BIO + UST-linear (no AQE)", metrics_linear_st, ref_metric=metrics_fixed_st)
            del sim_linear_st

        if hasattr(st_model, "reliability") and st_model.reliability is not None:
            sim_bio_sqrt_st, metrics_bio_sqrt_st, stats_sqrt = _fuse_and_eval(sim_vis, method="sqrt", return_stats=True)
            _log_perf("[2b]", "BIO + UST-sqrt (no AQE)", metrics_bio_sqrt_st, ref_metric=metrics_fixed_st)
        else:
            sim_bio_sqrt_st = sim_fixed_st
            metrics_bio_sqrt_st = metrics_fixed_st
            stats_sqrt = None

        if run_ls_baseline:
            sim_ls_st, metrics_ls_st, _ = _fuse_and_eval(sim_vis, fusion_rule_override="st_reid_ls")
            _log_perf("[2c]", "BIO + st-ReID-LS naive (debug)", metrics_ls_st, ref_metric=metrics_vis)
            del sim_ls_st

        # Current main relation-level fusion used by the downstream final pipeline.
        # By default this remains UST-sqrt. If yml explicitly changes it, it follows the yml.
        if _is_ls_fusion_rule(fusion_rule):
            sim_ust, metrics_ust, lambda_stats = _fuse_and_eval(sim_vis, fusion_rule_override="st_reid_ls", return_stats=False)
            fusion_type = "st-ReID-LS"
        elif use_uncertainty_aware and hasattr(st_model, "reliability") and st_model.reliability is not None:
            if str(uncertainty_method).lower() == "sqrt":
                sim_ust = sim_bio_sqrt_st
                metrics_ust = metrics_bio_sqrt_st
                lambda_stats = stats_sqrt
            else:
                sim_ust, metrics_ust, lambda_stats = _fuse_and_eval(sim_vis, method=uncertainty_method, return_stats=True)
            fusion_type = "Fixed-ST" if _is_fixed_st_method(uncertainty_method) else f"UST-{uncertainty_method}"
        else:
            sim_ust = sim_fixed_st
            metrics_ust = metrics_fixed_st
            fusion_type = "Fixed-ST"

        if lambda_stats is not None:
            logger.info(
                f"    adaptive lambda ({fusion_type}): mean={lambda_stats['mean']:.4f}, "
                f"std={lambda_stats['std']:.4f}, min={lambda_stats['min']:.4f}, max={lambda_stats['max']:.4f}"
            )
    else:
        logger.info("ST is disabled or unavailable. Detailed ST ablation skipped.")
        metrics_fixed_st = metrics_vis
        metrics_ust = metrics_vis
        fusion_type = "VIS"

    # -------------------------
    # 7) AQE / STG-AQE ablations and final result
    # -------------------------
    aqe_method = "No AQE"
    qf_aqe = qf

    # [3] BIO + Visual-AQE: no ST in candidate selection and no final ST calibration.
    if aqe_topk > 0 and aqe_alpha > 0:
        qf_bio_visual_aqe = apply_aqe_gpu(
            q_feats=qf,
            g_feats=gf,
            sim_mat=sim_vis,
            topk=aqe_topk,
            alpha=aqe_alpha,
        )
        dist_bio_visual_aqe = final_dist_blockwise(
            q_feats=qf_bio_visual_aqe,
            g_feats=gf,
            q_camids=q_camids,
            g_camids=g_camids,
            q_frames=q_frames,
            g_frames=g_frames,
            st_model=None,
            lambda_st=0.0,
            chunk_size=int(_cfg_get(cfg, "ST.BLOCKWISE_CHUNK_SIZE", 128)),
            devices=_resolve_blockwise_devices(cfg, fallback_device=device),
            out_dtype=torch.float16,
        )
        metrics_bio_visual_aqe = compute_per_query_metrics(dist_bio_visual_aqe, q_pids, g_pids, q_camids, g_camids)
        _log_perf("[3]", "BIO + Visual-AQE (no ST final)", metrics_bio_visual_aqe, ref_metric=metrics_vis)
        del qf_bio_visual_aqe, dist_bio_visual_aqe

    # [4] NoBIO + UST-sqrt + STG-AQE: isolates how much BIO contributes to the full pipeline.
    if (
        run_detailed_ablation
        and run_nobio_stgaqe_ablation
        and aqe_topk > 0
        and aqe_alpha > 0
        and use_st
        and st_model is not None
        and sim_nobio_sqrt_st_for_aqe is not None
    ):
        qf_nobio_stgaqe = apply_st_guided_aqe(
            q_feats=qf_original,
            g_feats=gf,
            sim_mat=sim_nobio_sqrt_st_for_aqe.float().to(device),
            q_cams=q_camids,
            q_frms=q_frames,
            g_cams=g_camids,
            g_frms=g_frames,
            st_model=st_model,
            topk=aqe_topk,
            alpha=aqe_alpha,
            use_reliability=True,
            eta0=0.5,
            tau=0.1,
        )
        dist_nobio_stgaqe = final_dist_blockwise(
            q_feats=qf_nobio_stgaqe,
            g_feats=gf,
            q_camids=q_camids,
            g_camids=g_camids,
            q_frames=q_frames,
            g_frames=g_frames,
            st_model=st_model,
            lambda_st=lambda_st,
            use_uncertainty_aware=True,
            uncertainty_method="sqrt",
            reliability_min=reliability_min,
            reliability_floor=reliability_floor,
            lambda_boost=lambda_boost,
            sigmoid_k=sigmoid_k,
            piecewise_thresholds=piecewise_thresholds,
            piecewise_weights=piecewise_weights,
            fusion_rule="weighted_sum",
            chunk_size=int(_cfg_get(cfg, "ST.BLOCKWISE_CHUNK_SIZE", 128)),
            devices=_resolve_blockwise_devices(cfg, fallback_device=device),
            out_dtype=torch.float16,
        )
        metrics_nobio_stgaqe = compute_per_query_metrics(dist_nobio_stgaqe, q_pids, g_pids, q_camids, g_camids)
        _log_perf("[4]", "NoBIO + UST-sqrt + STG-AQE", metrics_nobio_stgaqe, ref_metric=metrics_nobio_sqrt_st)
        del qf_nobio_stgaqe, dist_nobio_stgaqe

    # Main final path: BIO -> selected relation-level fusion -> STG-AQE -> final fusion.
    qf_after_bio = qf
    if aqe_topk > 0 and aqe_alpha > 0:
        if use_st_guided_aqe and use_st and st_model is not None:
            qf_aqe = apply_st_guided_aqe(
                q_feats=qf_after_bio,
                g_feats=gf,
                sim_mat=sim_ust.float().to(device),
                q_cams=q_camids,
                q_frms=q_frames,
                g_cams=g_camids,
                g_frms=g_frames,
                st_model=st_model,
                topk=aqe_topk,
                alpha=aqe_alpha,
                use_reliability=True,
                eta0=0.5,
                tau=0.1,
            )
            aqe_method = "ST-guided AQE"
        else:
            qf_aqe = apply_aqe_gpu(
                q_feats=qf_after_bio,
                g_feats=gf,
                sim_mat=sim_vis,
                topk=aqe_topk,
                alpha=aqe_alpha,
            )
            aqe_method = "Visual AQE"
    else:
        qf_aqe = qf_after_bio
        aqe_method = "No AQE"

    # Release large intermediate matrices before final blockwise scoring.
    if sim_nobio_sqrt_st_for_aqe is not None:
        del sim_nobio_sqrt_st_for_aqe
    for _name in ["sim_fixed_st", "sim_bio_sqrt_st", "sim_ust"]:
        if _name in locals() and locals()[_name] is not None:
            try:
                if _name != "sim_ust":
                    del locals()[_name]
            except Exception:
                pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_chunk_size = int(_cfg_get(cfg, "ST.BLOCKWISE_CHUNK_SIZE", 128))
    final_block_devices = _resolve_blockwise_devices(cfg, fallback_device=device)

    dist_final = final_dist_blockwise(
        q_feats=qf_aqe,
        g_feats=gf,
        q_camids=q_camids,
        g_camids=g_camids,
        q_frames=q_frames,
        g_frames=g_frames,
        st_model=st_model if (use_st and st_model is not None and (lambda_st > 0.0 or _is_ls_fusion_rule(fusion_rule))) else None,
        lambda_st=lambda_st,
        use_uncertainty_aware=use_uncertainty_aware,
        uncertainty_method=uncertainty_method,
        reliability_min=reliability_min,
        reliability_floor=reliability_floor,
        lambda_boost=lambda_boost,
        sigmoid_k=sigmoid_k,
        piecewise_thresholds=piecewise_thresholds,
        piecewise_weights=piecewise_weights,
        fusion_rule=fusion_rule,
        ls_lambda0=ls_lambda0,
        ls_gamma0=ls_gamma0,
        ls_lambda1=ls_lambda1,
        ls_gamma1=ls_gamma1,
        chunk_size=final_chunk_size,
        devices=final_block_devices,
        out_dtype=torch.float16,
    )
    metrics_final = compute_per_query_metrics(dist_final, q_pids, g_pids, q_camids, g_camids)

    mAP_final, final_r1, final_r5 = _metric_dict_to_map_r1_r5(metrics_final)
    cmc_final = np.zeros(5, dtype=np.float32)
    cmc_final[0] = final_r1
    cmc_final[4] = final_r5

    _log_perf("[5]", f"Final BIO + {fusion_type} + {aqe_method}", metrics_final, ref_metric=metrics_original)
    logger.info("=" * 92 + "\n")

    # -------------------------
    # 8) 可选 Re-ranking
    # -------------------------
    rerank_enable = bool(_cfg_get(cfg, "TEST.RERANK", False) or
                         _cfg_get(cfg, "TEST.RE_RANKING", False) or
                         _cfg_get(cfg, "TEST.RE_RANK", False))
    rerank_k1 = int(_cfg_get(cfg, "TEST.RERANK_K1", 20))
    rerank_k2 = int(_cfg_get(cfg, "TEST.RERANK_K2", 6))
    rerank_lambda = float(_cfg_get(cfg, "TEST.RERANK_LAMBDA", 0.3))
    rerank_block = int(_cfg_get(cfg, "TEST.RERANK_BLOCK", 1024))

    cmc_original_rr, mAP_original_rr = None, None
    cmc_final_rr, mAP_final_rr = None, None

    if rerank_enable:
        dist_gg = cosine_distmat_blockwise(gf, gf, block=rerank_block, out_dtype=torch.float16)

        qf_original_cpu = qf_original.detach().cpu()
        gf_cpu = gf.detach().cpu()
        qf_aqe_cpu = qf_aqe.detach().cpu()

        dist_qg_vis = cosine_distmat_blockwise(qf_original_cpu, gf_cpu, block=rerank_block, out_dtype=torch.float16)
        dist_qq_vis = cosine_distmat_blockwise(qf_original_cpu, qf_original_cpu, block=rerank_block, out_dtype=torch.float16)
        dist_vis_rr_np = k_reciprocal_re_ranking(
            dist_qg_vis, dist_qq_vis, dist_gg,
            k1=rerank_k1, k2=rerank_k2, lambda_value=rerank_lambda
        )
        metrics_original_rr = compute_per_query_metrics(
            torch.from_numpy(dist_vis_rr_np).float(), q_pids, g_pids, q_camids, g_camids
        )
        mAP_original_rr = float(np.nanmean(metrics_original_rr["ap"][metrics_original_rr["valid"]])) if metrics_original_rr["valid"].any() else 0.0
        cmc_original_rr = np.zeros(5, dtype=np.float32)
        cmc_original_rr[0] = float(np.nanmean(metrics_original_rr["rank1"][metrics_original_rr["valid"]])) if metrics_original_rr["valid"].any() else 0.0
        cmc_original_rr[4] = float(np.nanmean(metrics_original_rr["rank5"][metrics_original_rr["valid"]])) if metrics_original_rr["valid"].any() else 0.0

        dist_qg_final = dist_final.detach().to("cpu", dtype=torch.float16)
        dist_qq_final = cosine_distmat_blockwise(qf_aqe_cpu, qf_aqe_cpu, block=rerank_block, out_dtype=torch.float16)
        dist_final_rr_np = k_reciprocal_re_ranking(
            dist_qg_final, dist_qq_final, dist_gg,
            k1=rerank_k1, k2=rerank_k2, lambda_value=rerank_lambda
        )
        metrics_final_rr = compute_per_query_metrics(
            torch.from_numpy(dist_final_rr_np).float(), q_pids, g_pids, q_camids, g_camids
        )
        mAP_final_rr = float(np.nanmean(metrics_final_rr["ap"][metrics_final_rr["valid"]])) if metrics_final_rr["valid"].any() else 0.0
        cmc_final_rr = np.zeros(5, dtype=np.float32)
        cmc_final_rr[0] = float(np.nanmean(metrics_final_rr["rank1"][metrics_final_rr["valid"]])) if metrics_final_rr["valid"].any() else 0.0
        cmc_final_rr[4] = float(np.nanmean(metrics_final_rr["rank5"][metrics_final_rr["valid"]])) if metrics_final_rr["valid"].any() else 0.0

    # -------------------------
    # 9) 分桶实验
    # -------------------------
    grouped_rows_all = []
    per_query_records = []

    if run_grouped_analysis:
        valid_query_mask = metrics_original["valid"]

        # [A] d_q 分桶实验：最直接验证 BIO 是否更偏向困难 query
        dq_groups = build_tertile_groups(dq_scores.detach().cpu().numpy(), valid_mask=valid_query_mask)

        dq_rows = []
        dq_methods = [
            ("Baseline", metrics_original),
            ("BIO", metrics_vis),
            ("R2ID", metrics_final),
        ]
        for method_name, metric_dict in dq_methods:
            for group_name, group_mask in dq_groups["groups"].items():
                stat = summarize_metric_subset(metric_dict, group_mask)
                row = {
                    "experiment": "query_ambiguity",
                    "score_name": "d_q",
                    "threshold_q1": dq_groups["thresholds"][0],
                    "threshold_q2": dq_groups["thresholds"][1],
                    "method": method_name,
                    "group": group_name,
                    "count": stat["count"],
                    "mAP": stat["mAP"],
                    "R1": stat["R1"],
                    "R5": stat["R5"],
                }
                dq_rows.append(row)
                grouped_rows_all.append(row)

        log_grouped_table(
            logger,
            title="[A] Query ambiguity grouped experiment",
            score_name="Ambiguity",
            thresholds=dq_groups["thresholds"],
            rows=dq_rows,
        )

        # [B] R 分桶实验：验证 Fixed-ST / UST 在不同 camera-pair quality 上的表现
        if query_reliability_scores is not None and np.isfinite(query_reliability_scores).any():
            r_groups = build_tertile_groups(query_reliability_scores, valid_mask=valid_query_mask)
            r_rows = []
            r_methods = [("Baseline", metrics_original)]
            if metrics_nobio_fixed_st is not None:
                r_methods.append(("VIS+Fixed-ST", metrics_nobio_fixed_st))
            if metrics_nobio_linear_st is not None:
                r_methods.append(("VIS+UST-linear", metrics_nobio_linear_st))
            if metrics_nobio_sqrt_st is not None:
                r_methods.append(("VIS+UST-sqrt", metrics_nobio_sqrt_st))
            r_methods.append(("BIO", metrics_vis))
            r_methods.append(("BIO+Fixed-ST", metrics_fixed_st))
            if metrics_linear_st is not None:
                r_methods.append(("BIO+UST-linear", metrics_linear_st))
            if metrics_bio_sqrt_st is not None:
                r_methods.append(("BIO+UST-sqrt", metrics_bio_sqrt_st))
            if metrics_ls_st is not None:
                r_methods.append(("BIO+st-ReID-LS", metrics_ls_st))
            if metrics_bio_visual_aqe is not None:
                r_methods.append(("BIO+Visual-AQE", metrics_bio_visual_aqe))
            if metrics_nobio_stgaqe is not None:
                r_methods.append(("NoBIO+UST+STG-AQE", metrics_nobio_stgaqe))
            r_methods.append(("R2ID", metrics_final))
            for method_name, metric_dict in r_methods:
                for group_name, group_mask in r_groups["groups"].items():
                    stat = summarize_metric_subset(metric_dict, group_mask)
                    row = {
                        "experiment": "camera_pair_reliability",
                        "score_name": "R",
                        "threshold_q1": r_groups["thresholds"][0],
                        "threshold_q2": r_groups["thresholds"][1],
                        "method": method_name,
                        "group": group_name,
                        "count": stat["count"],
                        "mAP": stat["mAP"],
                        "R1": stat["R1"],
                        "R5": stat["R5"],
                    }
                    r_rows.append(row)
                    grouped_rows_all.append(row)

            log_grouped_table(
                logger,
                title="[B] Camera-pair reliability grouped experiment",
                score_name="Reliability",
                thresholds=r_groups["thresholds"],
                rows=r_rows,
            )

        # per-query 细表，后面论文画图会很有用
        dq_np = dq_scores.detach().cpu().numpy().astype(np.float32)
        q_rel_np = query_reliability_scores if query_reliability_scores is not None else np.full(len(q_pids), np.nan, dtype=np.float32)

        for i in range(len(q_pids)):
            rec = {
                "query_index": int(i),
                "pid": int(q_pids[i]),
                "camid": int(q_camids[i]),
                "frame": int(q_frames[i]) if len(q_frames) > i else -1,
                "valid": int(metrics_original["valid"][i]),
                "d_q": float(dq_np[i]) if np.isfinite(dq_np[i]) else float("nan"),
                "query_reliability": float(q_rel_np[i]) if np.isfinite(q_rel_np[i]) else float("nan"),
                "ap_baseline": float(metrics_original["ap"][i]) if np.isfinite(metrics_original["ap"][i]) else float("nan"),
                "r1_baseline": float(metrics_original["rank1"][i]) if np.isfinite(metrics_original["rank1"][i]) else float("nan"),
                "ap_vis_fixed_st": float(metrics_nobio_fixed_st["ap"][i]) if (metrics_nobio_fixed_st is not None and np.isfinite(metrics_nobio_fixed_st["ap"][i])) else float("nan"),
                "r1_vis_fixed_st": float(metrics_nobio_fixed_st["rank1"][i]) if (metrics_nobio_fixed_st is not None and np.isfinite(metrics_nobio_fixed_st["rank1"][i])) else float("nan"),
                "ap_vis_ust_linear": float(metrics_nobio_linear_st["ap"][i]) if (metrics_nobio_linear_st is not None and np.isfinite(metrics_nobio_linear_st["ap"][i])) else float("nan"),
                "r1_vis_ust_linear": float(metrics_nobio_linear_st["rank1"][i]) if (metrics_nobio_linear_st is not None and np.isfinite(metrics_nobio_linear_st["rank1"][i])) else float("nan"),
                "ap_vis_ust_sqrt": float(metrics_nobio_sqrt_st["ap"][i]) if (metrics_nobio_sqrt_st is not None and np.isfinite(metrics_nobio_sqrt_st["ap"][i])) else float("nan"),
                "r1_vis_ust_sqrt": float(metrics_nobio_sqrt_st["rank1"][i]) if (metrics_nobio_sqrt_st is not None and np.isfinite(metrics_nobio_sqrt_st["rank1"][i])) else float("nan"),
                "ap_bio": float(metrics_vis["ap"][i]) if np.isfinite(metrics_vis["ap"][i]) else float("nan"),
                "r1_bio": float(metrics_vis["rank1"][i]) if np.isfinite(metrics_vis["rank1"][i]) else float("nan"),
                "ap_fixed_st": float(metrics_fixed_st["ap"][i]) if np.isfinite(metrics_fixed_st["ap"][i]) else float("nan"),
                "r1_fixed_st": float(metrics_fixed_st["rank1"][i]) if np.isfinite(metrics_fixed_st["rank1"][i]) else float("nan"),
                "ap_ust_linear": float(metrics_linear_st["ap"][i]) if (metrics_linear_st is not None and np.isfinite(metrics_linear_st["ap"][i])) else float("nan"),
                "r1_ust_linear": float(metrics_linear_st["rank1"][i]) if (metrics_linear_st is not None and np.isfinite(metrics_linear_st["rank1"][i])) else float("nan"),
                "ap_st_reid_ls": float(metrics_ls_st["ap"][i]) if (metrics_ls_st is not None and np.isfinite(metrics_ls_st["ap"][i])) else float("nan"),
                "r1_st_reid_ls": float(metrics_ls_st["rank1"][i]) if (metrics_ls_st is not None and np.isfinite(metrics_ls_st["rank1"][i])) else float("nan"),
                "ap_ust": float(metrics_ust["ap"][i]) if np.isfinite(metrics_ust["ap"][i]) else float("nan"),
                "r1_ust": float(metrics_ust["rank1"][i]) if np.isfinite(metrics_ust["rank1"][i]) else float("nan"),
                "ap_bio_visual_aqe": float(metrics_bio_visual_aqe["ap"][i]) if (metrics_bio_visual_aqe is not None and np.isfinite(metrics_bio_visual_aqe["ap"][i])) else float("nan"),
                "r1_bio_visual_aqe": float(metrics_bio_visual_aqe["rank1"][i]) if (metrics_bio_visual_aqe is not None and np.isfinite(metrics_bio_visual_aqe["rank1"][i])) else float("nan"),
                "ap_nobio_stgaqe": float(metrics_nobio_stgaqe["ap"][i]) if (metrics_nobio_stgaqe is not None and np.isfinite(metrics_nobio_stgaqe["ap"][i])) else float("nan"),
                "r1_nobio_stgaqe": float(metrics_nobio_stgaqe["rank1"][i]) if (metrics_nobio_stgaqe is not None and np.isfinite(metrics_nobio_stgaqe["rank1"][i])) else float("nan"),
                "ap_r2id": float(metrics_final["ap"][i]) if np.isfinite(metrics_final["ap"][i]) else float("nan"),
                "r1_r2id": float(metrics_final["rank1"][i]) if np.isfinite(metrics_final["rank1"][i]) else float("nan"),
            }
            per_query_records.append(rec)

        if save_grouped_analysis:
            summary_csv = os.path.join(grouped_dir, "grouped_summary.csv")
            detail_csv = os.path.join(grouped_dir, "per_query_analysis.csv")
            save_grouped_rows_csv(summary_csv, grouped_rows_all)
            save_per_query_analysis_csv(detail_csv, per_query_records)
            logger.info(f"Grouped summary saved to: {summary_csv}")
            logger.info(f"Per-query analysis saved to: {detail_csv}")

    # -------------------------
    # 10) 结果汇总
    # -------------------------
    print("\n" + "=" * 104)
    print("FINAL PERFORMANCE SUMMARY (detailed ablation)")
    print("=" * 104)
    print(f"{'Method':<40} | {'mAP':>8} | {'Rank-1':>8} | {'Rank-5':>8} | {'ΔmAP':>8} | {'ΔR1':>8} | {'Ref':<18}")
    print("-" * 104)

    def _print_summary_row(name, metric_dict, ref_metric=None, ref_name=""):
        m, r1, r5 = _metric_dict_to_map_r1_r5(metric_dict)
        if ref_metric is None:
            dm, dr1 = None, None
            dm_s, dr1_s = "-", "-"
        else:
            rm, rr1, _ = _metric_dict_to_map_r1_r5(ref_metric)
            dm, dr1 = m - rm, r1 - rr1
            dm_s, dr1_s = f"{dm*100:+7.2f}%", f"{dr1*100:+7.2f}%"
        print(f"{name:<40} | {m*100:7.2f}% | {r1*100:7.2f}% | {r5*100:7.2f}% | {dm_s:>8} | {dr1_s:>8} | {ref_name:<18}")

    _print_summary_row("Original VIS", metrics_original)
    if cmc_original_rr is not None:
        print(f"{'Original VIS + ReRank':<40} | {mAP_original_rr*100:7.2f}% | {cmc_original_rr[0]*100:7.2f}% | {cmc_original_rr[4]*100:7.2f}% | {'-':>8} | {'-':>8} | {'-':<18}")

    # UST-only branch: BIO OFF, STG-AQE OFF.
    if metrics_nobio_fixed_st is not None:
        _print_summary_row("VIS + Fixed-ST", metrics_nobio_fixed_st, metrics_original, "Original VIS")
    if metrics_nobio_linear_st is not None:
        _print_summary_row("VIS + UST-linear", metrics_nobio_linear_st, metrics_nobio_fixed_st, "VIS+Fixed-ST")
    if metrics_nobio_sqrt_st is not None:
        _print_summary_row("VIS + UST-sqrt", metrics_nobio_sqrt_st, metrics_nobio_fixed_st, "VIS+Fixed-ST")
    if metrics_nobio_ls_st is not None:
        _print_summary_row("VIS + st-ReID-LS naive", metrics_nobio_ls_st, metrics_original, "Original VIS")

    # BIO branch.
    _print_summary_row("BIO + VIS", metrics_vis, metrics_original, "Original VIS")
    if metrics_fixed_st is not None:
        _print_summary_row("BIO + Fixed-ST", metrics_fixed_st, metrics_vis, "BIO+VIS")
    if metrics_linear_st is not None:
        _print_summary_row("BIO + UST-linear", metrics_linear_st, metrics_fixed_st, "BIO+Fixed-ST")
    if metrics_bio_sqrt_st is not None:
        _print_summary_row("BIO + UST-sqrt", metrics_bio_sqrt_st, metrics_fixed_st, "BIO+Fixed-ST")
    if metrics_ls_st is not None:
        _print_summary_row("BIO + st-ReID-LS naive", metrics_ls_st, metrics_vis, "BIO+VIS")
    if metrics_bio_visual_aqe is not None:
        _print_summary_row("BIO + Visual-AQE", metrics_bio_visual_aqe, metrics_vis, "BIO+VIS")
    if metrics_nobio_stgaqe is not None:
        _print_summary_row("NoBIO + UST-sqrt + STG-AQE", metrics_nobio_stgaqe, metrics_nobio_sqrt_st, "VIS+UST-sqrt")

    _print_summary_row(f"Final BIO + {fusion_type} + {aqe_method}", metrics_final, metrics_original, "Original VIS")

    if cmc_final_rr is not None:
        print(f"{'Final + ReRank':<40} | {mAP_final_rr*100:7.2f}% | {cmc_final_rr[0]*100:7.2f}% | {cmc_final_rr[4]*100:7.2f}% | {(mAP_final_rr-mAP_original_rr)*100:+7.2f}% | {(cmc_final_rr[0]-cmc_original_rr[0])*100:+7.2f}% | {'Orig+RR':<18}")
    print("=" * 104)

    logger.info(f"\nFINAL: Original VIS mAP={mAP_original:.2%}, Rank-1={cmc_original[0]:.2%}")
    logger.info(f"FINAL: R2ID mAP={mAP_final:.2%}, Rank-1={cmc_final[0]:.2%}")
    logger.info(f"FINAL: Gain mAP Δ{(mAP_final - mAP_original):+.2%}, Rank-1 Δ{(cmc_final[0] - cmc_original[0]):+.2%}")

    if bio_diagnostics:
        logger.info("BIO diagnostics:")
        for k, v in bio_diagnostics.items():
            logger.info(f"  {k}: {v}")

    rank1 = float(cmc_final[0]) if len(cmc_final) > 0 else 0.0
    rank5 = float(cmc_final[4]) if len(cmc_final) > 4 else 0.0
    return rank1, rank5
