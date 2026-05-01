## Added and Modified Files

- `utils/r2id_infer_plugin.py`: Implements the R2ID test-time enhancement pipeline, including BIO, UST-Fusion, STG-AQE, and final UST-Fusion.
- `utils/st_histogram_fusion.py`: Loads spatio-temporal histograms, camera-pair reliability matrices, and parses frame indices for query/gallery images.
- `tools/build_st_histogram.py`: Builds the spatio-temporal prior file in `.npz` format, including `distribution`, `reliability`, `interval`, and `num_bins`.
- `processor/processor_clipreid_r2id.py`: Provides an R2ID-enabled inference processor for the official CLIP-ReID Stage-2 pipeline.
- `test_clipreid_r2id.py`: Provides a standalone evaluation entry for R2ID-enhanced CLIP-ReID inference.
- `config/defaults.py`: Adds R2ID-related configuration nodes, including `R2ID`, `META`, `ST`, and `BIO`, to avoid unknown-key errors in YACS.

## 1. Build the Spatio-Temporal Histogram

The spatio-temporal histogram is built from the training set and saved as an `.npz` file. The generated file contains both the camera-pair temporal distribution and the corresponding reliability matrix.

### Market-1501

```bash
python tools/build_st_histogram.py \
  --train-dir /path/to/Market-1501-v15.09.15/bounding_box_train \
  --dataset market \
  --output-npz /path/to/st_market1501.npz \
  --num-bins 80 \
  --smooth-ratio 0.5
````

### DukeMTMC-reID / Occluded-Duke

```bash
python tools/build_st_histogram.py \
  --train-dir /path/to/Occluded_Duke/bounding_box_train \
  --dataset occ \
  --output-npz /path/to/st_occduke.npz \
  --num-bins 80 \
  --smooth-ratio 0.5
```

For DukeMTMC-reID, use:

```bash
python tools/build_st_histogram.py \
  --train-dir /path/to/DukeMTMC-reID/bounding_box_train \
  --dataset duke \
  --output-npz /path/to/st_dukemtmc.npz \
  --num-bins 80 \
  --smooth-ratio 0.5
```

### MSMT17

```bash
python tools/build_st_histogram.py \
  --train-dir /path/to/MSMT17/train \
  --dataset msmt17 \
  --output-npz /path/to/st_msmt17.npz \
  --num-bins 80 \
  --smooth-ratio 0.5
```

## 2. Run the Official CLIP-ReID Baseline

The original CLIP-ReID evaluation entry remains unchanged.

```bash
python test_clipreid.py \
  --config_file configs/person/cnn_clipreid.yml \
  TEST.WEIGHT /path/to/weight.pth
```

## 3. Run R2ID-Enhanced CLIP-ReID Inference

R2ID can be enabled through either command-line overrides or YAML configuration files.

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
  META.AQE.ALPHA 1.0 \
  BIO.ENHANCE.BASE_STRENGTH 0.5 \
  BIO.ENHANCE.POWER 1.1
```

The default R2ID inference order is:

```text
CLIP-ReID feature extraction
    -> BIO query correction
    -> UST-Fusion for candidate calibration
    -> STG-AQE query expansion
    -> final UST-Fusion
    -> evaluation
```

## 4. YAML-Based Evaluation

Example configuration files are provided under:

```text
configs/person/r2id/
```

A typical evaluation command is:

```bash
CUDA_VISIBLE_DEVICES=0 python test_clipreid_r2id.py \
  --config_file configs/person/r2id/market_rn50_r2id.yml
```

Before running, check the following fields in the YAML file:

```yaml
TEST:
  WEIGHT: /path/to/clipreid_checkpoint.pth

DATASETS:
  ROOT_DIR: /path/to/dataset/root

META:
  ST_HIST:
    NPZ_PATH: /path/to/st_histogram.npz
```

## 5. Ablation Settings

R2ID modules can be enabled or disabled through configuration overrides.

### BIO Only

```bash
python test_clipreid_r2id.py \
  --config_file configs/person/cnn_clipreid.yml \
  TEST.WEIGHT /path/to/weight.pth \
  R2ID.ENABLE True \
  META.ST_HIST.ENABLE False \
  BIO.ENHANCE.ENABLE True \
  META.AQE.ST_GUIDED False
```

### BIO + UST-Fusion

```bash
python test_clipreid_r2id.py \
  --config_file configs/person/cnn_clipreid.yml \
  TEST.WEIGHT /path/to/weight.pth \
  R2ID.ENABLE True \
  META.ST_HIST.ENABLE True \
  META.ST_HIST.NPZ_PATH /path/to/st.npz \
  META.AQE.ST_GUIDED False
```

### BIO + UST-Fusion + STG-AQE

```bash
python test_clipreid_r2id.py \
  --config_file configs/person/cnn_clipreid.yml \
  TEST.WEIGHT /path/to/weight.pth \
  R2ID.ENABLE True \
  META.ST_HIST.ENABLE True \
  META.ST_HIST.NPZ_PATH /path/to/st.npz \
  META.AQE.ST_GUIDED True
```

## 6. Recommended Main Configuration

The following settings correspond to the main R2ID test-time enhancement pipeline used in our experiments:

```yaml
R2ID:
  ENABLE: True

META:
  ST_HIST:
    ENABLE: True
    LAMBDA: 0.13
    FINAL_FUSION: True

  AQE:
    ST_GUIDED: True
    TOPK: 8
    ALPHA: 1.0
    TAU: 0.1
    RELIABILITY_SMOOTH: True
    ETA0: 0.5

BIO:
  ENHANCE:
    ENABLE: True
    TOPK: 6
    BASE_STRENGTH: 0.5
    POWER: 1.1
    TEMPERATURE: 0.05

ST:
  LAMBDA: 0.13
  UNCERTAINTY_METHOD: sqrt
  RELIABILITY_MIN: 0.4
  RELIABILITY_FLOOR: 0.1
  LAMBDA_BOOST: 1.0
  SIGMOID_K: 5.0
```


