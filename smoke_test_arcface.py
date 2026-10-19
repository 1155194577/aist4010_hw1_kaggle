"""Integration smoke test for the ArcFace/BNNeck change.

Builds a tiny 3-class dataset, trains a couple of epochs with each head, then
exercises checkpoint reload, TTA inference and submission writing.  Run it on a
machine with torch installed::

    python smoke_test_arcface.py
"""

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from config import Config, seed_everything  # noqa: E402
from dataset import get_test_loader, get_train_val_split  # noqa: E402
from engine import training_loop  # noqa: E402
from models import MarginHead, get_model  # noqa: E402
from predict import load_checkpoint, predict  # noqa: E402


def build_tiny_dataset(root: Path, classes=3, train_per_class=8, val_per_class=2, test_per=3):
    rng = np.random.RandomState(0)
    for split, per_class in (("train", train_per_class), ("val", val_per_class)):
        for c in range(classes):
            (root / split / f"id_{c}").mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                arr = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(arr).save(root / split / f"id_{c}" / f"{i:03d}.jpg")
    (root / "test").mkdir(parents=True, exist_ok=True)
    for i in range(classes * test_per):
        arr = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(root / "test" / f"test_{i}.jpg")
    with (root / "sample_submission.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "label"])
        for i in range(classes * test_per):
            writer.writerow([f"test_{i}.jpg", f"id_{i % classes}"])


def run_head(head: str, root: Path):
    cfg = Config(
        paths=root,
        train_dir=root / "train",
        val_dir=root / "val",
        test_dir=root / "test",
        checkpoint_path=root / f"ckpt_{head}.pt",
        epochs=2,
        batch_size=4,
        num_workers=0,
        resolution=64,
        head=head,
        margin=0.35,
        scale=30.0,
        bnneck=True,
        margin_warmup_epochs=1,
    )
    seed_everything(cfg.seed)
    train_loader, val_loader = get_train_val_split(cfg)
    num_classes = len(train_loader.dataset.classes)
    model = cfg.create_model(num_classes)
    assert model.head_type == head
    optimizer = cfg.create_optimizer(model)
    scheduler = cfg.create_scheduler(optimizer, len(train_loader))
    criterion = cfg.create_criterion()

    # Forward contract: labels optional, margin applied only when labels are given.
    images, labels = next(iter(train_loader))
    model.eval()  # avoid dropout randomness when comparing the two calls
    with torch.no_grad():
        plain = model(images)
        margined = model(images, labels)
    model.train()
    assert plain.shape == (images.size(0), num_classes), plain.shape
    assert torch.allclose(plain, margined, atol=1e-6) == (head == "linear")
    if head != "linear":
        target_idx = labels.view(-1, 1)
        changed = plain.gather(1, target_idx) - margined.gather(1, target_idx)
        assert (changed > 0).all(), "margin must lower the ground-truth logit"

    training_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        torch.device("cpu"),
        cfg,
        run=None,
        scheduler=scheduler,
    )

    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu")
    for key in ("head", "margin", "scale", "bnneck", "num_classes"):
        assert key in ckpt, f"checkpoint missing metadata: {key}"
    assert ckpt["head"] == head
    assert ckpt["num_classes"] == num_classes

    # Reload through the inference path (uses only checkpoint metadata).
    model, loaded_cfg, class_names, loaded_classes = load_checkpoint(
        cfg.checkpoint_path, Config(paths=root), torch.device("cpu")
    )
    assert loaded_cfg.head == head
    assert loaded_classes == num_classes
    assert class_names == train_loader.dataset.classes

    names, predictions, confidences = predict(
        model, get_test_loader(cfg), torch.device("cpu"), tta=True
    )
    assert len(names) == 9 and len(predictions) == 9
    assert all(0 <= p < num_classes for p in predictions)
    assert all(0.0 <= c <= 1.0 for c in confidences)
    print(
        f"[ok] head={head:<7} ckpt keys={sorted(k for k in ckpt if k != 'model_state')}"
    )


def margin_head_learns():
    """The margin head alone should separate fixed synthetic embeddings."""
    seed_everything(0)
    torch.manual_seed(0)
    num_classes, dim, per_class = 10, 32, 20
    centres = torch.randn(num_classes, dim)
    features = torch.cat(
        [centres[c] + 0.15 * torch.randn(per_class, dim) for c in range(num_classes)]
    )
    labels = torch.arange(num_classes).repeat_interleave(per_class)
    head = MarginHead(dim, num_classes, loss_type="arcface", margin=0.35, scale=30.0)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(150):
        optimizer.zero_grad()
        loss = criterion(head(features, labels), labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        accuracy = (head(features).argmax(1) == labels).float().mean().item()
    assert accuracy > 0.95, accuracy
    print(f"[ok] arcface head separates synthetic embeddings: acc={accuracy:.3f}")


def submission_cli(root: Path):
    import subprocess

    output = root / "submission.csv"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "predict.py"),
            "--checkpoint",
            str(root / "ckpt_arcface.pt"),
            str(root / "ckpt_cosface.pt"),
            "--test-dir",
            str(root / "test"),
            "--output",
            str(output),
            "--tta-scales",
            "1.0",
            "1.15",
            "--allow-collapse",
        ],
        check=True,
        cwd=REPO,
        capture_output=True,
    )
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 9, len(rows)
    assert all(row["id"].endswith(".jpg") for row in rows), rows[:3]
    assert list(rows[0]) == ["id", "label"]
    print("[ok] predict.py CLI writes sample-submission-compatible ids")


def resnet34_64px():
    model = get_model("resnet34", num_classes=1000, head="arcface")
    model.train()
    images = torch.randn(2, 3, 64, 64)
    labels = torch.randint(0, 1000, (2,))
    logits = model(images, labels)
    assert logits.shape == (2, 1000), logits.shape
    logits.sum().backward()
    assert model.embed_dim == 512
    with torch.no_grad():
        model.eval()
        assert model(images).shape == (2, 1000)
        assert model.embed(images).shape == (2, 512)
    print("[ok] resnet34 at 64 px: forward/backward, 512-d embedding")


def main():
    margin_head_learns()
    resnet34_64px()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_tiny_dataset(root)
        for head in ("arcface", "cosface", "linear"):
            run_head(head, root)
        submission_cli(root)
        # old plain-CE checkpoint must be rejected with a clear error
        legacy = root / "legacy.pt"
        torch.save({"model_state": {"fc.weight": torch.zeros(3, 512)}}, legacy)
        try:
            load_checkpoint(legacy, Config(paths=root), torch.device("cpu"))
        except ValueError as error:
            assert "old plain-CE head" in str(error)
            print("[ok] legacy checkpoint rejected with a helpful message")
        else:
            raise AssertionError("legacy checkpoint was not rejected")
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
