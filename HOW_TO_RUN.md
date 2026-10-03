# How to train and evaluate the 24 STR models

This checkout implements the original benchmark stages: `{None, TPS} × {VGG, RCNN, ResNet} × {None, BiLSTM} × {CTC, Attn}`. ConvNeXt and TCN in `docs/consult.md` are suggestions; they are not implemented here.

## 1. Set up data and Python

Use Python with PyTorch and a matching torchvision build. Install the remaining packages listed by the repository:

```bash
pip install lmdb pillow nltk natsort six numpy
```

Download the shared Dropbox folder as a zip (`dl=1` requests a download), then check and extract it:

```bash
wget -O ../dataset/data_lmdb_release.zip 'https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=1'
file ../dataset/data_lmdb_release.zip
unzip -t ../dataset/data_lmdb_release.zip
unzip -l ../dataset/data_lmdb_release.zip | head
unzip ../dataset/data_lmdb_release.zip -d ../dataset
```

If `unzip -t` fails, the archive is incomplete or invalid; download it again before training. A shared-folder download may contain another `data_lmdb_release.zip`; if so, extract that inner archive too. After extraction, make sure these paths exist (move the extracted folder if Dropbox added an extra directory level):

```text
../dataset/data_lmdb_release/
├── training/       # contains MJ and ST LMDB directories
├── validation/     # validation LMDB directories
└── evaluation/     # benchmark directories such as IIIT5k_3000, SVT, IC15_1811
```

Run all commands below from this repository directory after activating a Python environment with the dependencies installed. In this workspace, use `source ../.venv/bin/activate`. `dataset.py` uses the standard library's `itertools.accumulate` for compatibility with current PyTorch.

## 2. Smoke test (optional)

To smoke test both MJ and ST with the downloaded data, run one training iteration:

```bash
python3 train.py \
  --train_data ../dataset/data_lmdb_release/training \
  --valid_data ../dataset/data_lmdb_release/validation \
  --select_data MJ-ST --batch_ratio 0.5-0.5 \
  --Transformation None --FeatureExtraction VGG --SequenceModeling None --Prediction CTC \
  --exp_name smoke-mjst --batch_size 2 --num_iter 1 --valInterval 1 --workers 0
```

Expect `saved_models/smoke-mjst/best_accuracy.pth` and `best_norm_ED.pth`. This still scans all training labels and validates on the full validation set, so it can take time despite `--num_iter 1`.

## 3. Train all 24 combinations

The defaults are 300,000 iterations per run and batch size 192. This loop starts each model from random initialization and uses the MJ+ST training data. Run it on a CUDA machine; it creates one experiment directory per combination and saves the best validation checkpoints there.

Before starting the loop, check the extracted data and GPU:

```bash
ls ../dataset/data_lmdb_release/training ../dataset/data_lmdb_release/validation
python3 -c 'import torch; print("CUDA available:", torch.cuda.is_available())'
```

Stop if either data directory is missing or CUDA is unavailable.

```bash
for trans in None TPS; do
  for feat in VGG RCNN ResNet; do
    for seq in None BiLSTM; do
      for pred in CTC Attn; do
        CUDA_VISIBLE_DEVICES=0 python3 train.py \
          --train_data ../dataset/data_lmdb_release/training \
          --valid_data ../dataset/data_lmdb_release/validation \
          --select_data MJ-ST --batch_ratio 0.5-0.5 \
          --Transformation "$trans" --FeatureExtraction "$feat" \
          --SequenceModeling "$seq" --Prediction "$pred"
      done
    done
  done
done
```

The checkpoints are `saved_models/<Transformation>-<FeatureExtraction>-<SequenceModeling>-<Prediction>-Seed1111/best_accuracy.pth` and `best_norm_ED.pth`. Training all 24 is a substantial job: each of the 24 runs uses the 300,000-iteration default.

## 4. Evaluate one or all checkpoints

Example: evaluate the TPS-ResNet-BiLSTM-Attn checkpoint on all benchmark evaluation sets:

```bash
CUDA_VISIBLE_DEVICES=0 python3 test.py \
  --eval_data ../dataset/data_lmdb_release/evaluation --benchmark_all_eval \
  --Transformation TPS --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM --Prediction Attn \
  --saved_model saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth
```

To evaluate each of the 24 checkpoints, use the same grid and pass its stage names to `test.py`:

```bash
for trans in None TPS; do
  for feat in VGG RCNN ResNet; do
    for seq in None BiLSTM; do
      for pred in CTC Attn; do
        name="${trans}-${feat}-${seq}-${pred}-Seed1111"
        CUDA_VISIBLE_DEVICES=0 python3 test.py \
          --eval_data ../dataset/data_lmdb_release/evaluation --benchmark_all_eval \
          --Transformation "$trans" --FeatureExtraction "$feat" \
          --SequenceModeling "$seq" --Prediction "$pred" \
          --saved_model "saved_models/$name/best_accuracy.pth"
      done
    done
  done
done
```

Evaluation logs go under `result/`. The model stages and character settings supplied to `test.py` must match the checkpoint used for evaluation.
