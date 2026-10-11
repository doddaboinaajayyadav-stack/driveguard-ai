
from pathlib import Path
import random
import shutil

SOURCE = Path.home() / "Downloads" / "archive" / "train"
DEST = Path("dataset")

CLASSES = {
    "Open": "open",
    "Closed": "closed",
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
random.seed(42)

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Dataset source not found: {SOURCE}")

    # Avoid overwriting existing dataset images.
    for split in ("train", "val", "test"):
        for label in CLASSES.values():
            target = DEST / split / label
            if target.exists() and any(target.iterdir()):
                raise RuntimeError(
                    f"{target} already contains files. "
                    "Back them up before retrying."
                )

    for source_class, label in CLASSES.items():
        source_folder = SOURCE / source_class

        if not source_folder.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {source_folder}"
            )

        images = [
            p for p in source_folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]

        if not images:
            raise ValueError(f"No images found in {source_folder}")

        random.shuffle(images)

        n = len(images)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split, files in splits.items():
            target = DEST / split / label
            target.mkdir(parents=True, exist_ok=True)

            for index, image in enumerate(files):
                output = target / f"{label}_{index:04d}{image.suffix.lower()}"
                shutil.copy2(image, output)

            print(f"{split}/{label}: {len(files)} images")

    print("\nDataset preparation complete.")

if __name__ == "__main__":
    main()
