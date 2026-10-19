import argparse
import csv
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder

from config import Config
from dataset import get_test_loader, make_transform


def load_checkpoint(path, cfg, device):
    """
    Rebuild the model described by the checkpoint metadata and load its weights.
    """
    checkpoint = torch.load(path, map_location=device)
    meta = checkpoint if isinstance(checkpoint, dict) else {}
    state = meta.get("model_state", checkpoint)

    if "fc.weight" in state or "fc2.weight" in state:
        raise ValueError(
            "This checkpoint uses the old plain-CE head (fc/fc2 weights) and is not "
            "compatible with the ArcFace/BNNeck model. Retrain with the current code, "
            "or check out commit 21b779a to run inference on the old checkpoint."
        )

    # Head geometry is part of the architecture, not just a training choice, so it must
    # be restored from the checkpoint before building the model.
    for key, attr in (
        ("backbone", "backbone"),
        ("dropout", "dropout"),
        ("resolution", "resolution"),
        ("head", "head"),
        ("margin", "margin"),
        ("scale", "scale"),
        ("bnneck", "bnneck"),
    ):
        if key in meta:
            cfg = replace(cfg, **{attr: meta[key]})
    for key, attr in (("mean", "mean"), ("std", "std")):
        if key in meta:
            setattr(cfg, attr, tuple(meta[key]))
    class_names = meta.get("class_names")

    if class_names is None:
        print(
            "WARNING: checkpoint has no metadata; using Config defaults "
            f"(backbone={cfg.backbone}, head={cfg.head}, resolution={cfg.resolution}). "
            "Verify these match the training run or accuracy will silently collapse."
        )

    if "head.weight" in state:
        num_classes = state["head.weight"].shape[0]
    else:
        raise ValueError("Checkpoint does not contain a recognized classifier head.")

    model = cfg.create_model(num_classes)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg, class_names, num_classes


def predict(
    model,
    loader,
    device,
    tta,
):
    """
    this is the inference function, it will return the predictions, confidence and names of the images
    """
    predictions = []
    confidences = []
    names = []
    with torch.no_grad():
        for images, batch_names in loader:
            images = images.to(device)
            probabilities = torch.softmax(model(images), dim=1)
            if tta:
                flipped = torch.flip(images, dims=[3])
                probabilities = (
                    probabilities + torch.softmax(model(flipped), dim=1)
                ) / 2
            confidence, labels = probabilities.max(dim=1)
            predictions.extend(labels.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            names.extend(batch_names)
    return names, predictions, confidences


def verify(
    model,
    cfg,
    class_names,
    num_classes,
    device,
) -> None:
    val_dataset = ImageFolder(cfg.val_dir, transform=make_transform(cfg, train=False))
    if class_names is not None and val_dataset.classes != class_names:
        raise ValueError("Validation class ordering differs from checkpoint metadata.")
    if len(val_dataset.classes) != num_classes:
        raise ValueError("Validation class count differs from checkpoint classifier.")

    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    labels = []
    with torch.no_grad():
        for images, batch_labels in loader:
            labels.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())
    accuracy = sum(
        prediction == target
        for prediction, target in zip(labels, val_dataset.targets)
    ) / len(val_dataset)
    print(f"Validation accuracy: {accuracy:.4f}")


def validate_submission_vocabulary(output_labels, sample_submission_path: Path) -> None:
    if not sample_submission_path.exists():
        return

    with sample_submission_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["id", "label"]:
            raise ValueError(
                f"Sample submission must have columns ['id', 'label']; "
                f"found {reader.fieldnames}."
            )
        valid_labels = {row["label"] for row in reader}

    unknown_labels = set(output_labels) - valid_labels
    if unknown_labels:
        raise ValueError(
            f"Predictions contain labels not present in {sample_submission_path}: "
            f"{sorted(unknown_labels)[:5]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--allow-collapse", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    if args.test_dir is not None:
        cfg = replace(cfg, test_dir=args.test_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, class_names, num_classes = load_checkpoint(
        args.checkpoint, cfg, device
    )

    if class_names is None:
        class_names = ImageFolder(cfg.train_dir).classes
    if len(class_names) != num_classes:
        raise ValueError("Training class count differs from checkpoint classifier.")

    if args.verify:
        verify(model, cfg, class_names, num_classes, device)

    loader = get_test_loader(cfg)
    names, predictions, confidences = predict(model, loader, device, args.tta)
    if len(names) != len(loader.dataset):
        raise RuntimeError("Prediction count does not match test dataset size.")
    if any(label < 0 or label >= num_classes for label in predictions):
        raise RuntimeError("Prediction contains a label outside the classifier range.")
    output_labels = [class_names[label] for label in predictions]
    validate_submission_vocabulary(
        output_labels, cfg.paths / "sample_submission.csv"
    )
    counts = Counter(output_labels)
    if predictions:
        top_label, top_count = counts.most_common(1)[0]
        if top_count > 0.95 * len(predictions) and not args.allow_collapse:
            raise RuntimeError(
                f"{top_count}/{len(predictions)} predictions are class {top_label}; "
                "pass --allow-collapse to force."
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "label"])
        # Kaggle matches the `id` column against sample_submission.csv, whose ids keep
        # the file extension (test_123.jpg). Path(name).stem produced "test_123", so
        # no row matched and the submission was rejected / scored 0.
        writer.writerows(zip(names, output_labels))
    print(f"Mean confidence: {sum(confidences) / len(confidences):.4f}")
    print(f"Class distribution: {dict(sorted(counts.items()))}")
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()