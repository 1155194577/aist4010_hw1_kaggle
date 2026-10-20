import argparse
import csv
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
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
    # be restored from the checkpoint before building the model. ``last.pt`` stores the
    # full config under "cfg" instead of repeating the fields at the top level, so fall
    # back to it there.
    saved_cfg = meta.get("cfg") if isinstance(meta.get("cfg"), dict) else {}

    def checkpoint_value(key):
        return meta[key] if key in meta else saved_cfg.get(key)

    for key, attr in (
        ("backbone", "backbone"),
        ("dropout", "dropout"),
        ("resolution", "resolution"),
        ("head", "head"),
        ("margin", "margin"),
        ("scale", "scale"),
        ("bnneck", "bnneck"),
    ):
        value = checkpoint_value(key)
        if value is not None:
            cfg = replace(cfg, **{attr: value})
    for key, attr in (("mean", "mean"), ("std", "std")):
        value = checkpoint_value(key)
        if value is not None:
            setattr(cfg, attr, tuple(value))
    class_names = meta.get("class_names")

    if class_names is None:
        print(
            "WARNING: checkpoint has no class_names; using "
            f"backbone={cfg.backbone}, head={cfg.head}, resolution={cfg.resolution} "
            "and the training folder's class order."
        )

    if "head.weight" in state:
        num_classes = state["head.weight"].shape[0]
    else:
        raise ValueError("Checkpoint does not contain a recognized classifier head.")

    model = cfg.create_model(num_classes)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg, class_names, num_classes


def _resize(images, resolution):
    if tuple(images.shape[-2:]) == (resolution, resolution):
        return images
    return F.interpolate(
        images,
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def _tta_views(images, resolution, scales, flip):
    """Grow the image by ``scale``, centre-crop back, optionally hflip."""
    base = _resize(images, resolution)
    for scale in scales:
        if scale == 1.0:
            view = base
        else:
            enlarged = F.interpolate(
                base,
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            offset = (enlarged.shape[-1] - resolution) // 2
            view = enlarged[..., offset : offset + resolution, offset : offset + resolution]
        yield view
        if flip:
            yield torch.flip(view, dims=[3])


def predict(
    models,
    loader,
    device,
    tta=False,
    tta_scales=None,
    resolutions=None,
):
    """
    Average softmax probabilities over checkpoints and TTA views.

    ``models`` may be a single model (backwards compatible) or a list of models.
    ``tta_scales`` are zoom factors >= 1.0; each view is also horizontally flipped.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    if tta_scales is None:
        tta_scales = [1.0, 1.15] if tta else [1.0]
    tta_scales = [float(scale) for scale in tta_scales]
    if any(scale < 1.0 for scale in tta_scales):
        raise ValueError("TTA scales must be >= 1.0 (they zoom in before cropping).")
    flip = bool(tta or len(tta_scales) > 1)
    if resolutions is None:
        resolutions = [None] * len(models)

    for model in models:
        model.eval()

    views_per_model = len(tta_scales) * (2 if flip else 1)
    predictions = []
    confidences = []
    names = []
    with torch.no_grad():
        for images, batch_names in loader:
            images = images.to(device)
            probabilities = None
            for model, resolution in zip(models, resolutions):
                size = resolution or images.shape[-1]
                for view in _tta_views(images, size, tta_scales, flip):
                    probs = torch.softmax(model(view), dim=1)
                    probabilities = probs if probabilities is None else probabilities + probs
            probabilities = probabilities / (len(models) * views_per_model)
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
    tta=False,
    tta_scales=None,
) -> None:
    val_dataset = ImageFolder(cfg.val_dir, transform=make_transform(cfg, train=False))
    if class_names is not None and val_dataset.classes != class_names:
        raise ValueError("Validation class ordering differs from checkpoint metadata.")
    if len(val_dataset.classes) != num_classes:
        raise ValueError("Validation class count differs from checkpoint classifier.")

    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    scales = tta_scales if tta_scales is not None else ([1.0, 1.15] if tta else [1.0])
    flip = bool(tta or tta_scales)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            probabilities = None
            for view in _tta_views(images, cfg.resolution, scales, flip):
                probs = torch.softmax(model(view), dim=1)
                probabilities = probs if probabilities is None else probabilities + probs
            probabilities = probabilities / (len(scales) * (2 if flip else 1))
            predicted = probabilities.argmax(dim=1).cpu()
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)

    accuracy = correct / max(1, total)
    suffix = " (TTA)" if flip else ""
    print(f"Validation accuracy{suffix}: {accuracy:.4f}")


def validate_submission_vocabulary(
    output_labels, class_names, sample_submission_path: Path = None
) -> None:
    """Check predicted labels against the training classes.

    The sample submission's ``label`` column is only placeholder data: 2,394 rows
    drawn from 1,000 classes cover roughly 908 distinct names, so it must not be
    treated as the label vocabulary. The authoritative vocabulary is the
    checkpoint's ``class_names``.
    """
    valid_labels = set(class_names)
    unknown_labels = set(output_labels) - valid_labels
    if unknown_labels:
        raise ValueError(
            f"Predictions contain labels outside the {len(valid_labels)} training "
            f"classes: {sorted(unknown_labels)[:5]}"
        )

    if sample_submission_path is not None and sample_submission_path.exists():
        with sample_submission_path.open(newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != ["id", "label"]:
                raise ValueError(
                    f"Sample submission must have columns ['id', 'label']; "
                    f"found {reader.fieldnames}."
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        default=[Path("checkpoints/best_model.pt")],
        help="One or more checkpoints; probabilities are averaged (ensemble).",
    )
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--tta", action="store_true", help="hflip + 1.15x zoom TTA")
    parser.add_argument(
        "--tta-scales",
        type=float,
        nargs="+",
        default=None,
        help="Explicit zoom factors for TTA, e.g. 1.0 1.15 1.3 (implies --tta).",
    )
    parser.add_argument("--allow-collapse", action="store_true")
    args = parser.parse_args()

    base_cfg = Config()
    if args.test_dir is not None:
        base_cfg = replace(base_cfg, test_dir=args.test_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    cfgs = []
    class_names = None
    num_classes = None
    for checkpoint_path in args.checkpoint:
        model, model_cfg, names, classes = load_checkpoint(
            checkpoint_path, base_cfg, device
        )
        if not models:
            class_names, num_classes = names, classes
        elif names != class_names or classes != num_classes:
            raise ValueError(
                f"{checkpoint_path} uses a different class list or class count than "
                f"{args.checkpoint[0]}; only checkpoints of the same run family can "
                "be ensembled."
            )
        models.append(model)
        cfgs.append(model_cfg)
    if class_names is None:
        class_names = ImageFolder(cfgs[0].train_dir).classes
    if len(class_names) != num_classes:
        raise ValueError("Training class count differs from checkpoint classifier.")

    cfg = cfgs[0]
    scales = args.tta_scales if args.tta_scales is not None else ([1.0, 1.15] if args.tta else [1.0])
    print(
        f"Loaded {len(models)} checkpoint(s); TTA scales {scales}"
        + (" with hflip" if args.tta or args.tta_scales is not None else "")
    )

    if args.verify:
        verify(
            models[0], cfg, class_names, num_classes, device,
            tta=args.tta, tta_scales=args.tta_scales,
        )

    loader = get_test_loader(cfg)
    names, predictions, confidences = predict(
        models,
        loader,
        device,
        tta=args.tta,
        tta_scales=args.tta_scales,
        resolutions=[model_cfg.resolution for model_cfg in cfgs],
    )
    if len(names) != len(loader.dataset):
        raise RuntimeError("Prediction count does not match test dataset size.")
    if any(label < 0 or label >= num_classes for label in predictions):
        raise RuntimeError("Prediction contains a label outside the classifier range.")
    output_labels = [class_names[label] for label in predictions]
    validate_submission_vocabulary(
        output_labels, class_names, cfg.paths / "sample_submission.csv"
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