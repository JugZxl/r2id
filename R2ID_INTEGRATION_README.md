## 新增/修改文件

- `utils/r2id_infer_plugin.py`：BIO -> UST-Fusion -> STG-AQE -> final UST-Fusion 插件。
- `utils/st_histogram_fusion.py`：加载 ST 直方图、可靠性矩阵、解析 query/gallery 帧号。
- `tools/build_st_histogram.py`：由训练集构建 `distribution + reliability + interval + num_bins` 的 NPZ。
- `processor/processor_clipreid_r2id.py`：官方 Stage-2 推理的 R2ID 版本。
- `test_clipreid_r2id.py`：新增测试入口。
- `config/defaults.py`：只追加 R2ID/META/ST/BIO 配置节点，避免 yacs 报 unknown key。

## 1. 构建 ST 直方图

Market-1501 示例：

```bash
python tools/build_st_histogram.py \
  --train-dir /path/to/Market-1501-v15.09.15/bounding_box_train \
  --dataset market \
  --output-npz /path/to/st_market1501.npz \
  --num-bins 80 \
  --smooth-ratio 0.5
```

Occluded-Duke 示例：

```bash
python tools/build_st_histogram.py \
  --train-dir /path/to/dukemtmcreid/Occluded_Duke/bounding_box_train \
  --dataset occ \
  --output-npz /path/to/st_occduke.npz \
  --num-bins 80 \
  --smooth-ratio 0.5
```

MSMT17 示例：

```bash
python tools/build_st_histogram.py \
  --train-dir /path/to/MSMT17/train \
  --dataset msmt17 \
  --output-npz /path/to/st_msmt17.npz \
  --num-bins 80 \
  --smooth-ratio 0.5
```

## 2. 官方 CLIP-ReID 基线测试

原始入口仍可用：

```bash
python test_clipreid.py --config_file configs/person/cnn_clipreid.yml TEST.WEIGHT /path/to/weight.pth
```

## 3. R2ID 插件测试

```bash
python test_clipreid_r2id.py \
  --config_file configs/person/cnn_clipreid.yml \
  TEST.WEIGHT /path/to/weight.pth \
  DATASETS.NAMES market1501 \
  DATASETS.ROOT_DIR /path/to/data \
  R2ID.ENABLE True \
  META.ST_HIST.ENABLE True \
  META.ST_HIST.NPZ_PATH /path/to/st_market1501.npz \
  META.ST_HIST.LAMBDA 0.13 \
  ST.UNCERTAINTY_METHOD sqrt \
  META.AQE.TOPK 8 \
  META.AQE.ALPHA 0.8
```

## 4. 消融开关

只跑 BIO：

```bash
R2ID.ENABLE True META.ST_HIST.ENABLE False BIO.ENHANCE.ENABLE True META.AQE.ST_GUIDED False
```

BIO + UST，不跑 STG-AQE：

```bash
R2ID.ENABLE True META.ST_HIST.ENABLE True META.ST_HIST.NPZ_PATH /path/to/st.npz META.AQE.ST_GUIDED False
```

BIO + UST + STG-AQE：

```bash
R2ID.ENABLE True META.ST_HIST.ENABLE True META.ST_HIST.NPZ_PATH /path/to/st.npz META.AQE.ST_GUIDED True
```

