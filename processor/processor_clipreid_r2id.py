
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking
from utils.r2id_infer_plugin import R2IDInferencePlugin, cfg_get
from utils.st_histogram_fusion import parse_frame_ids


def _as_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() in {'true', '1', 'yes', 'y', 'on'}


def _feat_norm_enabled(cfg):
    return _as_bool(cfg_get(cfg, 'TEST.FEAT_NORM', True))


def _log_results(logger, title, cmc, mAP):
    logger.info(title)
    logger.info('mAP: {:.2%}'.format(mAP))
    for r in [1, 5, 10]:
        if len(cmc) >= r:
            logger.info('CMC curve, Rank-{:<3}:{:.2%}'.format(r, cmc[r - 1]))


def extract_clipreid_features(cfg, model, val_loader, device='cuda'):
    if device and torch.cuda.is_available():
        if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
    else:
        device = 'cpu'
        model.to(device)

    model.eval()
    feats = []
    pids = []
    camids_all = []
    img_paths = []

    with torch.no_grad():
        for img, pid, camid, camids, target_view, imgpath in val_loader:
            img = img.to(device, non_blocking=True)
            if cfg.MODEL.SIE_CAMERA:
                camids_input = camids.to(device, non_blocking=True)
            else:
                camids_input = None
            if cfg.MODEL.SIE_VIEW:
                target_view_input = target_view.to(device, non_blocking=True)
            else:
                target_view_input = None
            feat = model(img, cam_label=camids_input, view_label=target_view_input)
            feats.append(feat.detach().cpu())
            pids.extend(np.asarray(pid))
            camids_all.extend(np.asarray(camid))
            img_paths.extend(list(imgpath))

    feats = torch.cat(feats, dim=0)
    if _feat_norm_enabled(cfg):
        print('The test feature is normalized')
        feats = F.normalize(feats, dim=1, p=2)
    return feats, np.asarray(pids), np.asarray(camids_all), img_paths


def do_inference(cfg, model, val_loader, num_query):
    logger = logging.getLogger('transreid.test')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_r2id = _as_bool(cfg_get(cfg, 'R2ID.ENABLE', False))

    logger.info('Enter CLIP-ReID inference{}'.format(' with R2ID plugin' if use_r2id else ''))
    feats, pids, camids, img_paths = extract_clipreid_features(cfg, model, val_loader, device=device)

    qf = feats[:num_query]
    gf = feats[num_query:]
    q_pids = pids[:num_query]
    g_pids = pids[num_query:]
    q_camids = camids[:num_query]
    g_camids = camids[num_query:]

    if use_r2id:
        frames = parse_frame_ids(img_paths)
        q_frames = frames[:num_query]
        g_frames = frames[num_query:]
        logger.info('R2ID frame parsing done: Q={}, G={}, frame range=({}, {})'.format(
            len(q_frames), len(g_frames), int(frames.min()) if len(frames) else 0, int(frames.max()) if len(frames) else 0
        ))
        plugin = R2IDInferencePlugin.from_cfg(cfg, device=device)
        distmat_t, info = plugin(qf, gf, q_camids, g_camids, q_frames, g_frames)
        if info:
            logger.info('R2ID diagnostics: ' + ', '.join([f'{k}={v:.6f}' for k, v in sorted(info.items())]))
        distmat = distmat_t.float().numpy()
    elif _as_bool(cfg_get(cfg, 'TEST.RE_RANKING', False)):
        logger.info('=> Enter reranking')
        distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
    else:
        logger.info('=> Computing DistMat with euclidean_distance')
        distmat = euclidean_distance(qf, gf)

    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
    _log_results(logger, 'Validation Results', cmc, mAP)
    return cmc[0], cmc[4], mAP
