import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.model import DrowsinessCNN
from training.dataset_loader import EyeDataset


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    checkpoint_path = root / args.checkpoint
    test_dir = root / "dataset" / "test"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No trained checkpoint found: {checkpoint_path}. "
            "Train the model first."
        )

    dataset = EyeDataset(test_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model = DrowsinessCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Rows are actual labels; columns are predicted labels.
    confusion = torch.zeros((2, 2), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)
            for actual, predicted in zip(labels, predictions):
                confusion[int(actual), int(predicted)] += 1

    tn, fp = confusion[0, 0].item(), confusion[0, 1].item()
    fn, tp = confusion[1, 0].item(), confusion[1, 1].item()
    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    print(f"Test images: {total}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision (closed eyes): {precision:.4f}")
    print(f"Recall (closed eyes):    {recall:.4f}")
    print(f"F1 score (closed eyes):  {f1:.4f}")
    print("Confusion matrix (actual rows, predicted columns):")
    print(confusion.tolist())


if __name__ == "__main__":
    evaluate()
