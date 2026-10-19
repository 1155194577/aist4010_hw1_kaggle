# aist4010_hw1_kaggle

## Download competition data

Install the KaggleHub dependency and authenticate with Kaggle, then run:

```bash
.venv/bin/python download_data.py
```

The script copies the downloaded competition files into the repository's
`data/` directory, regardless of the directory from which the command is run.

## Weights & Biases tracking

Install the tracking dependency:

```bash
.venv/bin/pip install -r requirements.txt
```

Authenticate once without putting the API key in source control:

```bash
.venv/bin/wandb login
```

Run one training experiment:

```bash
.venv/bin/python train.py
```

Any `Config` field can be overridden from the command line, which is how seeds are
run without editing the file. Overriding `--seed`, `--backbone` or `--head` also
gives the run its own checkpoint (`checkpoints/<backbone>_<head>_seed<seed>.pt`)
so runs never overwrite each other:

```bash
.venv/bin/python train.py --seed 43
.venv/bin/python train.py --seed 44 --backbone resnet34
.venv/bin/python train.py --head cosface --epochs 80 --lr 1e-3 \
    --checkpoint checkpoints/cosface_e80.pt
```

Run the configured trials as separate W&B runs:

```bash
.venv/bin/python sweep.py
```

Training and validation loss are logged after every epoch. Each trial also
saves its best checkpoint under `checkpoints/`.

The default `CosineAnnealingWarmup` learning-rate scheduler warms up
linearly for one epoch, then decays the learning rate with a cosine schedule
over the remaining training steps. Its warmup duration can be changed in
`Config` with `warmup_epochs`.

Validation accuracy, weighted F1 score, and the classification report are
calculated after each epoch. Accuracy and F1 score are logged to W&B as
`val_accuracy` and `val_f1_score`.

To change the optimizer, update the `optimizer` field in `Config`:

```python
optimizer: str = "SGD"  # or "Adam", "AdamW", etc.
```

Both `resnet18` and `resnet34` are available as backbones and share the 64 px
stem (3x3 stride-1 first convolution, no max-pool, global average pooling), so
switching is a one-line change:

```python
cfg.backbone = "resnet34"
```

## Margin head (ArcFace / CosFace) and BNNeck

The classifier is now an embedding backbone plus a BNNeck (`BatchNorm1d` on the
embedding) and a margin head. The default is ArcFace, which replaces the plain
softmax cross-entropy objective:

```python
head: str = "arcface"        # "arcface", "cosface", or "linear"
margin: float = 0.35
scale: float = 30.0
bnneck: bool = True
margin_warmup_epochs: int = 2
ema_decay: float = 0.999      # exponential moving average of the weights; 0 disables
random_erasing: float = 0.25  # Cutout-style occlusion augmentation; 0 disables
```

`margin_warmup_epochs` ramps the margin in from zero so the first epochs are
stable; set it to `0` to use a constant margin. Label smoothing is applied only
to the `linear` head, because margin losses already regularise the logits.

The training loop passes labels to the model (`model(images, labels)`) so the
margin is applied to the ground-truth logit only; inference
(`model(images)`) returns `scale * cosine`. Checkpoints store `head`, `margin`,
`scale`, and `bnneck`, so `predict.py` rebuilds the right architecture
automatically. Checkpoints produced before this change use the old `fc` head and
are rejected with an explanatory error — retrain them with the current code.

Switching between heads is a one-line config change:

```python
cfg = Config()
cfg.head = "cosface"   # or "linear" to reproduce the old baseline
```

Generate a submission from the best checkpoint:

```bash
.venv/bin/python predict.py
```

Run validation checks against the checkpoint, or enable test-time augmentation:

```bash
.venv/bin/python predict.py --verify
.venv/bin/python predict.py --tta
.venv/bin/python predict.py --tta-scales 1.0 1.15 1.3
```

TTA averages probabilities over horizontal flips and zoom-then-centre-crop views.
Pass several checkpoints to ensemble them (they must share the same class list):

```bash
.venv/bin/python predict.py --checkpoint checkpoints/run_a.pt checkpoints/run_b.pt --tta
```

Ensembling 2-3 seeds is usually worth more than any single hyperparameter change.
When EMA is enabled the engine validates both the raw and the averaged weights and
keeps whichever is more accurate, so an early-epoch EMA can never overwrite a good
raw model.