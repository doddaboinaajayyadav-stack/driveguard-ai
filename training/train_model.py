import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import DrowsinessCNN
from training.dataset_loader import EyeDataset


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def train_model(epochs=20, batch_size=32, learning_rate=0.001):
    set_seed()

    project_root = Path(__file__).resolve().parents[1]
    train_dir = project_root / "dataset" / "train"
    val_dir = project_root / "dataset" / "val"
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load both datasets before starting training.
    train_data = EyeDataset(train_dir)
    val_data = EyeDataset(val_dir)

    if len(train_data) < 2 or len(val_data) < 1:
        raise ValueError(
            "Insufficient images. Add real images to dataset/train "
            "and dataset/val before training."
        )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=0
    )

    device = torch.device("cpu")
    model = DrowsinessCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    best_val_loss = float("inf")
    best_path = model_dir / "best_model.pt"
    patience = 5
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            count = labels.size(0)
            train_loss_sum += loss.item() * count
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_count += count

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device, dtype=torch.long)
                outputs = model(images)
                loss = criterion(outputs, labels)

                count = labels.size(0)
                val_loss_sum += loss.item() * count
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_count += count

        train_loss = train_loss_sum / train_count
        train_accuracy = train_correct / train_count
        val_loss = val_loss_sum / val_count
        val_accuracy = val_correct / val_count

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_accuracy={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_accuracy={val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_index": {"open": 0, "closed": 1},
                    "image_size": 24,
                    "architecture": "DrowsinessCNN",
                },
                best_path,
            )
            print(f"Saved best checkpoint: {best_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping: validation loss did not improve.")
            break

    print("Training finished. Best checkpoint:", best_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the DriveGuard AI eye classifier."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("Training parameters must be positive.")

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
