# How to train and evaluate the 24 STR models

This checkout implements the original benchmark stages: `{None, TPS} × {VGG, RCNN, ResNet} × {None, BiLSTM} × {CTC, Attn}`. ConvNeXt and TCN in `docs/consult.md` are suggestions; they are not implemented here.

## 1. Set up data and Python

Use Python with PyTorch and a matching torchvision build. Install the remaining packages listed by the repository:

```bash
pip install lmdb pillow nltk natsort six numpy
```

Download and extract the [benchmark LMDB data](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) here so these paths exist:

```text
data_lmdb_release/
├── training/       # contains MJ and ST LMDB directories
├── validation/     # validation LMDB directories
└── evaluation/     # benchmark directories such as IIIT5k_3000, SVT, IC15_1811
```

Run all commands below from this repository directory. The current workspace has a virtual environment one directory above; use `../.venv/bin/python` in place of `python3` if using it. `dataset.py` uses the standard library's `itertools.accumulate` for compatibility with current PyTorch.

## 2. Smoke test (optional)

This workspace includes a small plate LMDB that can verify training and checkpoint writing. It is only a pipeline check, not scene-text pretraining:

```bash
python3 train.py \
  --train_data ../satria-data/lmdb/train --valid_data ../satria-data/lmdb/val \
  --select_data / --batch_ratio 1 \
  --Transformation None --FeatureExtraction VGG --SequenceModeling None --Prediction CTC \
  --exp_name smoke-ctc --batch_size 2 --num_iter 1 --valInterval 1 --workers 0
```

Expect `saved_models/smoke-ctc/best_accuracy.pth` and `best_norm_ED.pth`.

## 3. Train all 24 combinations

The defaults are 300,000 iterations per run and batch size 192. This loop starts each model from random initialization and uses the MJ+ST training data. Run it on a CUDA machine; it creates one experiment directory per combination and saves the best validation checkpoints there.

```bash
for trans in None TPS; do
  for feat in VGG RCNN ResNet; do
    for seq in None BiLSTM; do
      for pred in CTC Attn; do
        CUDA_VISIBLE_DEVICES=0 python3 train.py \
          --train_data data_lmdb_release/training \
          --valid_data data_lmdb_release/validation \
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
  --eval_data data_lmdb_release/evaluation --benchmark_all_eval \
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
          --eval_data data_lmdb_release/evaluation --benchmark_all_eval \
          --Transformation "$trans" --FeatureExtraction "$feat" \
          --SequenceModeling "$seq" --Prediction "$pred" \
          --saved_model "saved_models/$name/best_accuracy.pth"
      done
    done
  done
done
```

Evaluation logs go under `result/`. The model stages and character settings supplied to `test.py` must match the checkpoint used for evaluation.
