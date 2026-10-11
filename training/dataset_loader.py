from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_TO_INDEX = {"open": 0, "closed": 1}


class EyeDataset(Dataset):
    """Load labeled eye images from open/ and closed/ folders."""

    def __init__(self, root, image_size=24):
        self.root = Path(root)
        self.image_size = image_size
        self.samples = []

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.root}"
            )

        for class_name, label in CLASS_TO_INDEX.items():
            class_dir = self.root / class_name

            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing class directory: {class_dir}"
                )

            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, label))

        if not self.samples:
            raise ValueError(
                f"No supported images found in {self.root}. "
                "Add real images to open/ and closed/."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("L")
            image = image.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BILINEAR,
            )
            pixels = torch.tensor(
                list(image.getdata()), dtype=torch.float32
            ).reshape(1, self.image_size, self.image_size) / 255.0

        return pixels, label
