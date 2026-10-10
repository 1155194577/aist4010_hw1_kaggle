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

Run the configured trials as separate W&B runs:

```bash
.venv/bin/python sweep.py
```

Training and validation loss are logged after every epoch. Each trial also
saves its best checkpoint under `checkpoints/`.

Validation accuracy, weighted F1 score, and the classification report are
calculated after each epoch. Accuracy and F1 score are logged to W&B as
`val_accuracy` and `val_f1_score`.