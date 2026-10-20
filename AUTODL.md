# Running on AutoDL — adapted from the "省钱绝招" guide

AutoDL's cost-saving page describes three scenarios. Here is how each maps to
this repository.

| AutoDL advice | What it means for us |
|---|---|
| **无卡模式** (no-GPU boot, ¥0.1/h) | use it for uploading data, `pip install`, code edits and running the CPU smoke test. Never train there. |
| **升降配置** (change GPU count) | debug with 1 GPU and `--epochs 1`, then run the full schedule. This code is single-GPU; more GPUs need DDP, so scale the *schedule*, not the GPU count. |
| **`/usr/bin/shutdown` after the job** | `train.py --shutdown` powers the instance off when training ends. Logs must go to a file, because stdout disappears on shutdown. |

## 1. Create the instance (no GPU yet)

- Image: a PyTorch image matching the GPU (e.g. PyTorch 2.x, CUDA 12.1).
- Boot in **无卡模式** — it costs ¥0.1/h and releases the GPU.

Everything heavy lives on the data disk:

```bash
mkdir -p /root/autodl-tmp/aist4010
cd /root/autodl-tmp/aist4010
git clone https://github.com/1155194577/aist4010_hw1_kaggle.git
cd aist4010_hw1_kaggle
pip install -r requirements.txt          # torch is already in the image
```

Upload `data.zip` (JupyterLab upload or `scp`; see AutoDL's upload guide), then:

```bash
unzip -q ~/data.zip -d .                 # creates ./data/train, val, test
python -c "import torch; print(torch.__version__)"
```

Save the image now (*保存镜像* in the console) so the next instance starts with
the environment ready.

## 2. Boot with a GPU and smoke-test

```bash
cd /root/autodl-tmp/aist4010/aist4010_hw1_kaggle
python train.py --seed 43 --epochs 1 \
    --checkpoint /root/autodl-tmp/aist4010/ckpt/debug.pt
```

One epoch is ~1–2 minutes and validates data loading, ArcFace and checkpoints.

## 3. Full run with auto-shutdown

Checkpoints and logs must live on `/root/autodl-tmp`, and the command can always
use `--resume last`: if no checkpoint exists it starts fresh, if the instance was
interrupted it continues.

```bash
cd /root/autodl-tmp/aist4010/aist4010_hw1_kaggle
mkdir -p /root/autodl-tmp/aist4010/ckpt

nohup python train.py --seed 43 --epochs 64 \
    --checkpoint /root/autodl-tmp/aist4010/ckpt/seed43.pt \
    --resume last \
    --shutdown \
    > /root/autodl-tmp/aist4010/train_seed43.log 2>&1 &

tail -f /root/autodl-tmp/aist4010/train_seed43.log
```

`--shutdown` calls `/usr/bin/shutdown` from a `finally` block, so the instance
powers off even if training raises an error — a crashed job must not keep
billing. Consequences:

- checkpoints are already on disk (`*_last.pt` after every epoch, `best_model`
  whenever validation improves);
- the log survives because it was redirected to a file;
- **without `--shutdown` the process simply exits and the instance keeps
  running (and billing)**. You then stop it manually in the AutoDL console.

If the run is interrupted (spot instance, manual stop), start the **same command**
again — `--resume last` loads `seed43_last.pt` and continues at the right epoch,
including optimizer and scheduler state.

> Keep `--epochs` identical when resuming. The LR schedule is built from
> `epochs`; resuming with a different value keeps the old schedule position.
> To extend a finished run, start a new run instead of resuming.

## 4. Cost sketch

| Step | Time | Approx. |
|---|---|---|
| 无卡模式: upload data, install deps, debug | ~1 h | ¥0.1 |
| 1-epoch GPU smoke test | 2 min | < ¥0.2 |
| 64 epochs, ResNet-18 @64 px, RTX 4090 | 40–60 min | ¥1.5–3 |
| 200 epochs, IR-50 (the next step) | 4–6 h | ¥10–18 |

Compare with the university cluster: free, but the home quota is 2.4 GB, which
cannot hold PyTorch plus the dataset — AutoDL's data disk removes that limit.

## 5. Notes

- W&B needs login (`wandb login`) or `export WANDB_MODE=offline`; offline runs can
  be uploaded later with `wandb sync wandb/offline-run-*`.
- The data disk `/root/autodl-tmp` is the large one; the system disk is ~30 GB.
- After the first successful setup, *保存镜像* turns the environment into a
  reusable image, so new instances need no `pip install`.
