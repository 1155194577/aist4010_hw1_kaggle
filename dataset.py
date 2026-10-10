from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config import Config


def make_transform(cfg: Config, train: bool = True) -> transforms.Compose:
    transform_list = []
    if train and cfg.augmentation:
        transform_list.extend(
            [
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(), ## not for face classification
                transforms.RandomRotation(degrees=15, fill=(128, 128, 128)),
                transforms.RandomResizedCrop(
                    cfg.resolution,
                    scale=cfg.crop_scale,
                    ratio=cfg.crop_ratio,
                ),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                ),
            ]
        )
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ]
    )
    RandomErasing = transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
    transform_list.append(RandomErasing)
    return transforms.Compose(transform_list)


def visualize_augmentation(
    image_path: Path | str,
    cfg: Config,
    output_path: Path | str = "augmentation_preview.jpg",
    samples: int = 8,
) -> Path:
    if samples < 1:
        raise ValueError("samples must be at least 1")

    source = Image.open(image_path).convert("RGB")
    transform = make_transform(cfg, train=True)
    to_image = transforms.ToPILImage()
    mean = torch.tensor(cfg.mean).view(3, 1, 1)
    std = torch.tensor(cfg.std).view(3, 1, 1)

    def restore_colors(image: torch.Tensor) -> Image.Image:
        image = (image.cpu() * std + mean).clamp(0, 1)
        return to_image(image)

    images = [source.resize((cfg.resolution, cfg.resolution))]
    images.extend(restore_colors(transform(source)) for _ in range(samples))

    columns = min(4, len(images))
    rows = (len(images) + columns - 1) // columns
    grid = Image.new("RGB", (columns * cfg.resolution, rows * cfg.resolution), "white")
    for index, image in enumerate(images):
        x = (index % columns) * cfg.resolution
        y = (index // columns) * cfg.resolution
        grid.paste(image, (x, y))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)
    return output


if __name__ == "__main__":
    cfg = Config()
    image_path = next(Path(cfg.paths).glob("train/*/*.jpg"))
    output_path = visualize_augmentation(image_path, cfg)
    print(f"Saved augmentation preview to {output_path}")