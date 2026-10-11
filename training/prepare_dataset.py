from pathlib import Path
import random
import shutil

SOURCE = Path.home() / "Downloads" / "archive" / "train"
DEST = Path("dataset")
CLASSES = {"Open": "open", "Closed": "closed"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE}")

    for split in ("train", "val", "test"):
        for label in CLASSES.values():
            folder = DEST / split / label
            if folder.exists() and any(folder.iterdir()):
                raise RuntimeError(f"Folder is not empty: {folder}")

    random.seed(42)

    for source_name, label in CLASSES.items():
        source_folder = SOURCE / source_name
        images = [
            p for p in source_folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]

        if not images:
            raise ValueError(f"No images found in {source_folder}")

        random.shuffle(images)
        n = len(images)
        splits = {
            "train": images[:int(n * 0.70)],
            "val": images[int(n * 0.70):int(n * 0.85)],
            "test": images[int(n * 0.85):],
        }

        for split, files in splits.items():
            target = DEST / split / label
            target.mkdir(parents=True, exist_ok=True)

            for i, image in enumerate(files):
                shutil.copy2(image, target / f"{label}_{i:04d}{image.suffix.lower()}")

            print(f"{split}/{label}: {len(files)} images")

    print("Dataset preparation complete.")

if __name__ == "__main__":
    main()
